# Profile Networks Report (`ptflops`)

## Command Executed
```bash
(swiftnet) ale@potentia:~/work/av-tse/SEANet$ python scripts/profile_networks_ptflops.py \
    --models av_sepformer seanet dprnn muse \
    --duration 1 \
    --sample-rate 16000 \
    --fps 25 \
    --visual-frontend pretrain_networks/visual_frontend.pt
```

---

## Global Setup / Visual Frontend Summary

* **Frontend:** DeepAVSR Visual Frontend
* **Frontend Parameters:** 11,185,088 (11.185 M)
* **Frontend GMACs / sec:** 12.6722
* **Frontend GMACs / frame avg:** 0.5069

---

## Summary Comparison Table

| Model | Backend Params | Total Params | Backend GMACs/s | Frontend GMACs/s | Total GMACs/s | VF Compute Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AV_SEPFORMER** | 21.734 M | 32.919 M | 77.2505 | 12.6722 | 89.9227 | 14.09% |
| **SEANET** | 8.700 M | 19.886 M | 14.4275 | 12.6722 | 27.0997 | 46.76% |
| **DPRNN** | 4.119 M | 15.304 M | 4.9088 | 12.6722 | 17.5810 | 72.08% |
| **MUSE** | 15.014 M | 26.199 M | 8.6499 | 12.6722 | 21.3221 | 59.43% |

---

## Detailed Results by Model

### 1. AV_SEPFORMER

#### Configuration
| Property | Value |
| :--- | :--- |
| **Duration** | 1.000 s |
| **Sample Rate** | 16,000 Hz |
| **Audio Samples** | 16,000 |
| **Visual FPS** | 25.0 |
| **Visual Frames** | 25 |
| **Audio Latent Steps** | 799 |
| **Visual/Audio Ratio** | 31.96 |

#### Performance Breakdown
* **Backend Parameters:** 21,734,022 (21.734 M)
* **AV-TSE Backend GMACs/s:** 77.2505
* **End-to-End Total Params:** 32,919,110 (32.919 M)
* **End-to-End Total GMACs/s:** 89.9227
* **Visual Frontend Compute Share:** **14.09%**

---

### 2. SEANET

#### Configuration
| Property | Value |
| :--- | :--- |
| **Duration** | 1.000 s |
| **Sample Rate** | 16,000 Hz |
| **Audio Samples** | 16,000 |
| **Visual FPS** | 25.0 |
| **Visual Frames** | 25 |
| **Audio Latent Steps** | 799 |
| **Visual/Audio Ratio** | 31.96 |

#### Performance Breakdown
* **Backend Parameters:** 8,700,422 (8.700 M)
* **AV-TSE Backend GMACs/s:** 14.4275
* **End-to-End Total Params:** 19,885,510 (19.886 M)
* **End-to-End Total GMACs/s:** 27.0997
* **Visual Frontend Compute Share:** **46.76%**

---

### 3. DPRNN

#### Configuration
| Property | Value |
| :--- | :--- |
| **Duration** | 1.000 s |
| **Sample Rate** | 16,000 Hz |
| **Audio Samples** | 16,000 |
| **Visual FPS** | 25.0 |
| **Visual Frames** | 25 |
| **Audio Latent Steps** | 799 |
| **Visual/Audio Ratio** | 31.96 |

#### Performance Breakdown
* **Backend Parameters:** 4,119,302 (4.119 M)
* **AV-TSE Backend GMACs/s:** 4.9088
* **End-to-End Total Params:** 15,304,390 (15.304 M)
* **End-to-End Total GMACs/s:** 17.5810
* **Visual Frontend Compute Share:** **72.08%**

---

### 4. MUSE

#### Configuration
| Property | Value |
| :--- | :--- |
| **Duration** | 1.000 s |
| **Sample Rate** | 16,000 Hz |
| **Audio Samples** | 16,000 |
| **Visual FPS** | 25.0 |
| **Visual Frames** | 25 |
| **Audio Latent Steps** | 799 |
| **Visual/Audio Ratio** | 31.96 |

#### Performance Breakdown
* **Backend Parameters:** 15,014,109 (15.014 M)
* **AV-TSE Backend GMACs/s:** 8.6499
* **End-to-End Total Params:** 26,199,197 (26.199 M)
* **End-to-End Total GMACs/s:** 21.3221
* **Visual Frontend Compute Share:** **59.43%**
