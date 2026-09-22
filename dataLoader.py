import os
import random

import numpy as np
import soundfile
import torch


# ============================================================
# Constants
# ============================================================

SAMPLE_RATE = 16000


# ============================================================
# DataLoaders
# ============================================================

def init_loader(args):

    args.trainLoader = torch.utils.data.DataLoader(
        train_loader(
            set_type="train",
            **vars(args),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )

    args.valLoader = torch.utils.data.DataLoader(
        train_loader(
            set_type="val",
            **vars(args),
        ),
        batch_size=8,
        shuffle=False,
        num_workers=args.n_cpu,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )

    args.testLoader = torch.utils.data.DataLoader(
        test_loader(
            **vars(args),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return args


# ============================================================
# Audio loading
# ============================================================

def load_audio(path, length):
    """
    Load an audio utterance and force it to the duration specified
    in data_list.csv.

    Args:
        path:
            Path to .wav.

        length:
            Utterance duration in seconds.

    Returns:
        np.ndarray [T]
    """

    audio, _ = soundfile.read(path)

    max_audio = int(round(length * SAMPLE_RATE))

    if audio.shape[0] < max_audio:
        shortage = max_audio - audio.shape[0]

        # Preserve original SEANet behaviour.
        audio = np.pad(
            audio,
            (0, shortage),
            mode="wrap",
        )

    audio = audio[:max_audio]

    return audio


# ============================================================
# Visual loading
# ============================================================

def load_visual(path):
    """
    Load precomputed visual embeddings.

    IMPORTANT:
        The embeddings have already been generated at the desired
        visual FPS (25, 20, 17.5, 15, 12.5, ...).

        Therefore we DO NOT resize, interpolate, crop or pad the
        temporal dimension here.

    Expected shape:
        [T_visual, D]

    For SEANet:
        D = 512
    """

    face = np.load(path)

    if face.ndim != 2:
        raise ValueError(
            f"Expected visual embeddings with shape [T, D], "
            f"but got {face.shape} for:\n{path}"
        )

    if face.shape[0] == 0:
        raise ValueError(
            f"Empty visual embedding file:\n{path}"
        )

    return face


# ============================================================
# Mixture generation
# ============================================================

def audio_overlap(
    label,
    infer1,
    snr1,
    infer2,
    snr2,
    noise,
    snrn,
    addition_speaker,
    addition_noise,
):
    """
    Generate the audio mixture.

    Behaviour intentionally kept equivalent to the original
    SEANet loader.
    """

    label_db = 10 * np.log10(
        np.mean(label ** 2) + 1e-4
    )

    infer1_db = 10 * np.log10(
        np.mean(infer1 ** 2) + 1e-4
    )

    infer1 = (
        np.sqrt(
            10 ** (
                (label_db - infer1_db - snr1) / 10
            )
        )
        * infer1
    )

    audio = label + infer1

    # --------------------------------------------------------
    # Optional second interfering speaker
    # --------------------------------------------------------

    if addition_speaker:

        infer2_db = 10 * np.log10(
            np.mean(infer2 ** 2) + 1e-4
        )

        infer2 = (
            np.sqrt(
                10 ** (
                    (label_db - infer2_db - snr2) / 10
                )
            )
            * infer2
        )

        audio = audio + infer2

    # --------------------------------------------------------
    # Optional additive noise
    # --------------------------------------------------------

    if addition_noise:

        noise_db = 10 * np.log10(
            np.mean(noise ** 2) + 1e-4
        )

        noise = (
            np.sqrt(
                10 ** (
                    (label_db - noise_db - snrn) / 10
                )
            )
            * noise
        )

        audio = audio + noise

    return audio


# ============================================================
# Safe normalization
# ============================================================

def peak_normalize(x):
    """
    Preserve the original SEANet peak-normalization behaviour,
    but avoid division by zero for silent signals.
    """

    peak = np.max(np.abs(x))

    if peak > 0:
        x = x / peak

    return x


# ============================================================
# Synchronized audio / visual crop
# ============================================================

def synchronized_crop(
    audio,
    label,
    noise,
    face,
    crop_length,
    visual_fps,
):
    """
    Randomly crop audio and visual embeddings while preserving
    temporal synchronization.

    The crop starting point is selected on the VISUAL timeline.

    Example at 15 FPS:

        start_face = 12

        start_time = 12 / 15
                   = 0.8 s

        start_audio = round(0.8 * 16000)
                    = 12800

    This also works correctly for fractional FPS such as
    17.5 and 12.5.

    The maximum starting frame is constrained by BOTH:
        1. available visual frames
        2. available audio samples

    Therefore the returned crop always has fixed dimensions.
    """

    visual_fps = float(visual_fps)

    if visual_fps <= 0:
        raise ValueError(
            f"visual_fps must be > 0, got {visual_fps}"
        )

    # --------------------------------------------------------
    # Required crop sizes
    # --------------------------------------------------------

    audio_crop_length = int(
        round(crop_length * SAMPLE_RATE)
    )

    visual_crop_length = int(
        round(crop_length * visual_fps)
    )

    if visual_crop_length <= 0:
        raise ValueError(
            f"Invalid visual crop length: "
            f"{visual_crop_length} frames "
            f"(length={crop_length}, fps={visual_fps})"
        )

    # --------------------------------------------------------
    # Maximum start allowed by visual sequence
    # --------------------------------------------------------

    max_start_face_visual = (
        face.shape[0] - visual_crop_length
    )

    # --------------------------------------------------------
    # Maximum start allowed by audio sequence
    # --------------------------------------------------------

    max_start_audio = min(
        audio.shape[0],
        label.shape[0],
        noise.shape[0],
    ) - audio_crop_length

    # --------------------------------------------------------
    # Check that a complete crop exists
    # --------------------------------------------------------

    if max_start_face_visual < 0:
        raise RuntimeError(
            "Visual utterance is shorter than requested crop:\n"
            f"  available frames : {face.shape[0]}\n"
            f"  required frames  : {visual_crop_length}\n"
            f"  visual_fps       : {visual_fps}\n"
            f"  crop length      : {crop_length} s"
        )

    if max_start_audio < 0:
        raise RuntimeError(
            "Audio utterance is shorter than requested crop:\n"
            f"  available samples: {audio.shape[0]}\n"
            f"  required samples : {audio_crop_length}\n"
            f"  crop length      : {crop_length} s"
        )

    # --------------------------------------------------------
    # Convert maximum AUDIO start into visual-frame units.
    #
    # We use floor because this is an upper bound:
    # selecting a larger visual frame could map past the last
    # valid audio starting sample.
    # --------------------------------------------------------

    max_start_face_audio = int(
        np.floor(
            (max_start_audio / SAMPLE_RATE)
            * visual_fps
        )
    )

    # --------------------------------------------------------
    # Must be valid in BOTH modalities
    # --------------------------------------------------------

    max_start_face = min(
        max_start_face_visual,
        max_start_face_audio,
    )

    max_start_face = max(
        0,
        max_start_face,
    )

    # --------------------------------------------------------
    # Random visual starting frame
    # --------------------------------------------------------

    if max_start_face > 0:
        start_face = random.randint(
            0,
            max_start_face,
        )
    else:
        start_face = 0

    # --------------------------------------------------------
    # Visual frame -> timestamp -> audio sample
    # --------------------------------------------------------

    start_time = (
        start_face / visual_fps
    )

    start_audio = int(
        round(
            start_time * SAMPLE_RATE
        )
    )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    audio_crop = audio[
        start_audio:
        start_audio + audio_crop_length
    ]

    label_crop = label[
        start_audio:
        start_audio + audio_crop_length
    ]

    noise_crop = noise[
        start_audio:
        start_audio + audio_crop_length
    ]

    face_crop = face[
        start_face:
        start_face + visual_crop_length,
        :
    ]

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if audio_crop.shape[0] != audio_crop_length:
        raise RuntimeError(
            "Audio crop has incorrect length:\n"
            f"  got             : {audio_crop.shape[0]}\n"
            f"  expected        : {audio_crop_length}\n"
            f"  start_audio     : {start_audio}\n"
            f"  start_face      : {start_face}\n"
            f"  visual_fps      : {visual_fps}\n"
            f"  full audio      : {audio.shape[0]}\n"
            f"  full visual     : {face.shape[0]}"
        )

    if label_crop.shape[0] != audio_crop_length:
        raise RuntimeError(
            f"Label crop wrong: "
            f"{label_crop.shape[0]} != {audio_crop_length}"
        )

    if noise_crop.shape[0] != audio_crop_length:
        raise RuntimeError(
            f"Noise crop wrong: "
            f"{noise_crop.shape[0]} != {audio_crop_length}"
        )

    if face_crop.shape[0] != visual_crop_length:
        raise RuntimeError(
            "Visual crop has incorrect length:\n"
            f"  got             : {face_crop.shape[0]}\n"
            f"  expected        : {visual_crop_length}\n"
            f"  start_face      : {start_face}\n"
            f"  visual_fps      : {visual_fps}\n"
            f"  full visual     : {face.shape[0]}"
        )

    return (
        audio_crop,
        label_crop,
        noise_crop,
        face_crop,
    )


# ============================================================
# Train / validation dataset
# ============================================================

class train_loader(object):

    def __init__(
        self,
        set_type,
        data_list,
        visual_path,
        audio_path,
        length,
        musan_path=None,
        visual_fps=25.0,
        **kwargs,
    ):
        self.set_type = set_type

        self.visual_path = visual_path
        self.audio_path = audio_path

        self.length = float(length)
        self.visual_fps = float(visual_fps)

        self.data_list = []

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        with open(data_list, "r") as f:
            lines = f.read().splitlines()

        # ----------------------------------------------------
        # Create speaker -> class ID mapping
        # ----------------------------------------------------

        speaker_keys = []

        for line in lines:

            data = line.split(",")

            if data[0] == set_type:
                speaker_keys.append(
                    data[2]
                )

        speaker_keys = sorted(
            set(speaker_keys)
        )

        speaker_dict = {
            speaker: index
            for index, speaker
            in enumerate(speaker_keys)
        }

        # ----------------------------------------------------
        # Parse entries
        # ----------------------------------------------------

        for line in lines:

            data = line.split(",")

            if data[0] != set_type:
                continue

            speaker_label = speaker_dict[
                data[2]
            ]

            label_name = os.path.join(
                data[2],
                data[3],
            )

            inter_name1 = os.path.join(
                data[6],
                data[7],
            )

            snr1 = round(
                float(data[8]),
                3,
            )

            all_length = float(
                data[-1]
            )

            self.data_list.append(
                [
                    label_name,
                    inter_name1,
                    snr1,
                    all_length,
                    speaker_label,
                ]
            )

        print(
            f"{set_type}: "
            f"{len(self.data_list)} samples | "
            f"visual_fps={self.visual_fps}"
        )

    def __getitem__(self, index):

        (
            label_name,
            inter_name1,
            snr1,
            all_length,
            speaker_id,
        ) = self.data_list[index]

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        label_path = os.path.join(
            self.audio_path,
            label_name + ".wav",
        )

        inter_path = os.path.join(
            self.audio_path,
            inter_name1 + ".wav",
        )

        visual_path = os.path.join(
            self.visual_path,
            label_name + ".npy",
        )

        # ----------------------------------------------------
        # Load target / interferer / visual embeddings
        # ----------------------------------------------------

        label = load_audio(
            path=label_path,
            length=all_length,
        )

        inter_1 = load_audio(
            path=inter_path,
            length=all_length,
        )

        # IMPORTANT:
        # Use the actual temporal length of the generated .npy.
        face = load_visual(
            path=visual_path,
        )

        # ----------------------------------------------------
        # Generate mixture
        # ----------------------------------------------------

        inter_2 = None
        snr2 = None

        noise = None
        snrn = None

        audio = audio_overlap(
            label=label,
            infer1=inter_1,
            snr1=snr1,
            infer2=inter_2,
            snr2=snr2,
            noise=noise,
            snrn=snrn,
            addition_speaker=False,
            addition_noise=False,
        )

        # Interference component
        noise = audio - label

        # ----------------------------------------------------
        # Synchronized random crop
        # ----------------------------------------------------

        try:
            (
                audio,
                label,
                noise,
                face,
            ) = synchronized_crop(
                audio=audio,
                label=label,
                noise=noise,
                face=face,
                crop_length=self.length,
                visual_fps=self.visual_fps,
            )

        except Exception as e:
            raise RuntimeError(
                f"\nFailed processing sample:\n"
                f"  index        : {index}\n"
                f"  set          : {self.set_type}\n"
                f"  target       : {label_name}\n"
                f"  interferer   : {inter_name1}\n"
                f"  CSV duration : {all_length}\n"
                f"  visual FPS   : {self.visual_fps}\n"
                f"  visual file  : {visual_path}\n"
                f"Reason:\n{e}"
            ) from e

        # ----------------------------------------------------
        # Original SEANet normalization
        # ----------------------------------------------------

        audio = peak_normalize(audio)
        label = peak_normalize(label)
        noise = peak_normalize(noise)

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        return (
            torch.from_numpy(
                np.asarray(
                    audio,
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(
                    face,
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(
                    label,
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(
                    noise,
                    dtype=np.float32,
                )
            ),
            speaker_id,
        )

    def __len__(self):
        return len(
            self.data_list
        )


# ============================================================
# Test dataset
# ============================================================

class test_loader(object):

    def __init__(
        self,
        data_list,
        audio_path,
        visual_path,
        musan_path=None,
        visual_fps=25.0,
        **kwargs,
    ):
        self.audio_path = audio_path
        self.visual_path = visual_path

        self.visual_fps = float(
            visual_fps
        )

        self.data_list = []

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        with open(data_list, "r") as f:
            lines = f.read().splitlines()

        # ----------------------------------------------------
        # Test entries
        # ----------------------------------------------------

        for line in lines:

            data = line.split(",")

            if data[0] != "test":
                continue

            label_name = os.path.join(
                data[2],
                data[3],
            )

            inter_name1 = os.path.join(
                data[6],
                data[7],
            )

            snr1 = round(
                float(data[8]),
                3,
            )

            all_length = float(
                data[-1]
            )

            self.data_list.append(
                [
                    label_name,
                    inter_name1,
                    snr1,
                    all_length,
                ]
            )

        print(
            f"test: "
            f"{len(self.data_list)} samples | "
            f"visual_fps={self.visual_fps}"
        )

    def __getitem__(self, index):

        (
            label_name,
            inter_name1,
            snr1,
            all_length,
        ) = self.data_list[index]

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        label_path = os.path.join(
            self.audio_path,
            label_name + ".wav",
        )

        inter_path = os.path.join(
            self.audio_path,
            inter_name1 + ".wav",
        )

        visual_path = os.path.join(
            self.visual_path,
            label_name + ".npy",
        )

        # ----------------------------------------------------
        # Load complete utterances
        # ----------------------------------------------------

        label = load_audio(
            path=label_path,
            length=all_length,
        )

        inter_1 = load_audio(
            path=inter_path,
            length=all_length,
        )

        # Keep the actual number of generated visual embeddings.
        face = load_visual(
            path=visual_path,
        )

        # ----------------------------------------------------
        # Mixture
        # ----------------------------------------------------

        audio = audio_overlap(
            label=label,
            infer1=inter_1,
            snr1=snr1,
            infer2=None,
            snr2=None,
            noise=None,
            snrn=None,
            addition_speaker=False,
            addition_noise=False,
        )

        noise = audio - label

        # ----------------------------------------------------
        # Original SEANet normalization
        # ----------------------------------------------------

        audio = peak_normalize(audio)
        label = peak_normalize(label)
        noise = peak_normalize(noise)

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        return (
            torch.from_numpy(
                np.asarray(
                    audio,
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(
                    face,
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(
                    label,
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(
                    noise,
                    dtype=np.float32,
                )
            ),
            torch.from_numpy(
                np.asarray(
                    noise,
                    dtype=np.float32,
                )
            ),
        )

    def __len__(self):
        return len(
            self.data_list
        )