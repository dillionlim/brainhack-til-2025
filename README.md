# DSTA BrainHack TIL-AI 2025 - Team DeepSheet

<!-- TOC -->
* [Introduction](#introduction)
* [Team Members (in alphabetical order)](#team-members-in-alphabetical-order)
* [Achievements](#achievements)
* [Final evaluation results](#final-evaluation-results)
* [ASR](#asr)
  * [Exploratory Data Analysis](#exploratory-data-analysis-eda)
  * [Denoising](#denoising)
  * [Training](#training)
    * [Training Attempt(Whisper, Denoising)](#training-attempt-whisper-denoising)
    * [Training (Parakeet)](#training-parakeet)
  * [Hyperparameters](#hyperparameters)
  * [Inference](#inference)
  * [Additional Steps to Improve](#additional-steps-to-improve)
* [CV](#cv)
* [OCR](#ocr)
* [RL](#rl)
* [Surprise Task](#surprise)
* [Hardware used](#hardware-used)
* [Final words](#final-words)
<!-- TOC -->

## Introduction
TIL-AI 2025 comprised of 3 standard tasks, 1 surprise task (introduced on day 1 of the semi-finals, with approximately 12 hours to complete the task) and an overarching reinforcement learning (RL) task:
* **Automatic speech recognition (ASR)** \
    Convert noisy, accented audio into a text transcript.
* **Compute Vision (CV)** \
    Detect and classify (sometimes) small objects in a noisy image, with classes chosen from a known, target list.
* **Optical Character Recognition (OCR)** \
    Transcribe text from a noisy image of a document with varied fonts and layouts.
* **Shredded Document Reconstruction (Surprise Task)** \
    Reorder image slices of equal width from a shredded text document into their original order. 

For reinforcement learning (RL), the aim was to drive a TurtleBot3 Burger through a simulated maze environment, interacting with other agents and completing challenges.

Each match is played by four teams, and consists of four rounds, such that each team will play a scout once, and guards three times [^1]. 

As a scout, the aim is to:

* Avoid capture by the guards (-500 points),
* Collect reconnaissance points (80% distribution) placed around the map (+10 point), and
* Complete challenges located at mission points in the map (up to +50 points for semi-finals and finals, guaranteed +50 points for qualifiers).

At mission points (for semi-finals and finals), a scout will receive 5 missions, chosen from the above 4 tasks with equal probability. You can receive any where from 0 to 10 points for each task, corresponding to the accuracy of your model for the given task.

As a guard, the aim is to capture the scout (+500 points).

[^1]: The points mentioned are 10 times that used in qualifiers, and the abovementioned point distribution was used for semi-finals and finals.

### Additional Note

When using this repository, you will need to initialize git submodules, using 

```
git submodule update --init
```

## Team Members (in alphabetical order)
* [Jiang Xinnan](https://github.com/jxinnan): RL
* [Lim Dillion](https://github.com/dillionlim): ASR / RL / Surprise
* [Ng See Jay](https://github.com/CJBuzz): OCR / RL
* [Xie Kai Wen](https://github.com/XieKaiwen): OCR / RL / Surprise
* [Zhou Hao Ren](https://github.com/haoren-zhou): CV / Surprise

## Achievements
* 1st overall in Qualifiers
* 2nd placing in 1st seeding group in semi-finals
* 1579 points (first run - official score) / ~1700 points (second run - after restart due to hardware restart) in semi-finals (which puts us at 4th - 5th placing in terms of raw score across the entire Advanced category)
* Surprise Challenge Winner

In this format, a single round was conducted to decide the winner in each bracket.

We theorize that our subpar semi-finals showing was due to the fact that:

1. 2 other teams were not actively moving about, which contributes to a harder time catching the guard as a scout-guard pair moving optimally will result in the scout never getting caught.
2. A slight bug in the observation layer for the guard model ended up resetting the entire observation layer, leading to oscillations observed in 1 round as a guard.
3. We prioritised a low-risk strategy, which involved us tuning for the "best worst case" (maximising the minimum, reducing the variance) and playing more conservatively. The matchup ended up facing off against a riskier scout, which paid off with less aggressive guards. This conservative scout assumes that all 3 guards will be actively searching for the scout, and therefore avoids high-risk, unknown situations (elaborated below).

## Evaluation results (Qualifiers)
| Task | Model | Accuracy score | Speed Score |
|-|-|-|-|
| ASR | parakeet-tdt-0.6b-v2 | 0.952 | 0.939 |
| CV | TODO | 0.532 | 0.947 |
| OCR | PaddleOCR [^2] | 1.000 | 0.941 |

## Evaluation results (Pre-semi-finals)
| Task | Model | Accuracy score | Speed Score |
|-|-|-|-|
| ASR | parakeet-tdt-0.6b-v2 (Unchanged) | 0.952 | 0.939 |
| CV | TODO | 0.615 | 0.892 [^3] |
| OCR | DocTR | 0.981 | 0.841 |
| Surprise | TSP-SSIM (Low-N-Regime) + Beam Search-SSIM (High-N-Regime) [^4] | 1.000 | 0.965 |

It is interesting to note that the reinforcement learning scores will not be introduced here, because the variance in scores would be very high between different submissions. Strategies to mitigate this during qualifiers and semi-finals/finals respectively would be discussed in the RL section.

[^2]: This was achieved via an unconventional technique, which will be elaborated on, below, hence, the semi-final model would be significantly different.
[^3]: This was achieved using hardware-specific TensorRT optimizations. However, since semi-final participant servers were built on a 5070Ti architecture, which we do not have access to, we ended up switching to a Pytorch implementation with a slightly lower, but comparable speed score.
[^4]: C++ and Python implementations ended up having the exact same accuracy / speed combination, to 3 significant figures.

## ASR

As with all previous iterations of BrainHack TIL-AI, ASR has traditionally always been one of the given tasks. However, the ASR task this year, commendably, seems to be significantly noisier, leading to a wider spread of scores.

### Exploratory Data Analysis (EDA)

Sampling a few `.wav` files from the dataset, it was noted that the input was significantly noisier.

We shall look at 2 different samples that we used for EDA to understand the samples that we are working with.

![Spectrogram (Sample 3890)](docs/asr/sample_3890_spectrogram.png)
*Sample 3890 Spectrogram (sample_3890.wav in advanced dataset)*

This case was (unfortunately) one of the first samples that we looked at, and it is likely on the harder side of samples provided.

As we can see, there is a high-frequency noise in the spectrogram (corresponding to a rooster's crowing), which entirely drowns out the speech to the human ear. Unfortunately, this high-frequency noise also overlaps with the speaking frequency (which seems to have a larger range of frequencies than normaly expected, likely due to the background noise added in).

![Spectrogram (Sample 4269)](docs/asr/sample_4269_spectrogram.png)
*Sample 4269 Spectrogram (sample_4269.wav in advanced dataset)*

This is a more representative sample of the normal input speech, which just consists of a speech with a distinct, narrow band of frequencies, and a large range of background noise, which is much weaker in intensity compared to the main speech. 

### Denoising

Based on the above, there was first an attempt done to get the normal range of frequency of human voices to do aggressive denoising. According to [research](https://www.researchgate.net/publication/235008404_Studying_audition_in_fossil_hominins_A_new_approach_to_the_evolution_of_language?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoiX2RpcmVjdCJ9fQ), the [normal human speech frequency range for intelligibility lies between 300 to 3000 Hz](https://www.researchgate.net/figure/The-modern-human-audiogram-and-speech-frequencies-The-solid-line-represents-the-minimum_fig4_235008404), but the fundamental frequency for males typically ranges from 90 to 155 Hz, while for females, it ranges from 165 to 255 Hz. However, from quick testing, it appears that normal speech tends to lie up to 1400 Hz (if the dataset is non-adverse).

There were two main ways that were used to do denoising, the first being [spectral gating with the `noisereduce` library implementation, default hyperparameters](https://github.com/timsainb/noisereduce) and the second being a low-pass filter over the dataset, using a straightforward FFT implementation. 

Spectral gating was the less aggressive approach, but failed to denoise instances such as sample 3890. We then attempted to do a low-pass filter over the input. While it managed to get rid of the high-frequency noise, it gets rid of a lot of critical information since the speech band largely overlaps with the high-frequency noise, which hurts the training, even though the speech is now audible to a human ear. 

This can be seen in the FFT spectrum of samples 3890 and 4490, where there cannot be a clean cutoff between noise and speech.

![FFT_3890](docs/asr/sample_3890_fft.png)
*Sample 3890 FFT (sample_3890.wav in advanced dataset)*

![FFT_4490](docs/asr/sample_4490_fft.png)
*Sample 4490 FFT (sample_4490.wav in advanced dataset)*

This would be clear when we ended up training it on a denoised dataset.

### Training

#### Training Attempt (Whisper, Denoising)

We tried denoising the given data, then training on the original noisy dataset combined with denoised data. However, the model ended up not performing very well, as with last year. In fact, the denoised data performed worse.

Other than that, Whisper did not show much promise, only scoring slightly less than 0.9 with finetuning, and would likely score in the low 0.9s even with more hyperparameter tuning and training.

| Model | Hyperparameters and Information | Score | Speed |
|-|-|-|-|
| whisper-base.en | First 4000 samples (no denoised), 2000 steps, denoising before evaluation | 0.218  | 0.285  | 
| whisper-base.en | First 4000 samples | 0.873 | 0.857 |
| whisper-base.en | 4500 samples + denoised | 0.852 | 0.855 |
| whisper-small | 4500 samples, 2000 steps, no denoising | 0.899  | 0.594  |
| whisper-small.en | First 4000 samples, K-Fold cross validation, K = 5 | 0.895  | 0.617  |

At this point, we noticed that Whisper was not doing as well as expected, so we went onto the HuggingFace Open ASR leaderboard.

Some relevant comparisons that we looked at are reproduced below for convenience:

| Model | Average WER | RTFx | 
| - | - | - |
| nvidia/parakeet-tdt-0.6b-v2 | 6.05 | 3386.02 |
| distil-whisper/distil-large-v3.5 | 7.21 | 202.03 |
| openai/whisper-large-v3 | 7.44 | 145.51 |
| openai/whisper-small.en | 8.59 | 268.91 |

Other ASR Models in the NVIDIA parakeet family have been omitted, because at that point, Parakeet TDT 0.6B V2 had both the lowest average WER and the highest RTFx [^5].

[^5] At the point of writing this writeup, there has been a new model, ibm-granite/granite-speech-3.3-8b which has an average WER of 5.85, which is lower than the model used. However, the RTFx is 31.33, which is about 100 times lower than that of the chosen Parakeet model. This model was released on May 16 2025, but we did not use it when we noticed it before the semi-finals because of the low RTFx, given that the 25% speed score was quite significant. 

#### Training (Parakeet)

Based on initial training on Parakeet, it was fairly robust to noise, we decided not to proceed with any further augmentations. Also, drawing from above experiences, we made the decision to not train with denoising.

The preliminary testing for hyperparameter (epoch only) tuning were:

| Hyperparameters and Information | Score | Speed |
|-|-|-|
| First 4000 samples | 10 epochs raw | 0.936  | 0.929 |
| First 4000 samples, batch inference | 10 epochs raw | 0.936  | 0.936 |
| First 4000 samples for 10 epochs, 4500 samples for 2 more epochs | 0.914 | 0.94 |
| First 4000 samples for 10 epochs, 4500 samples for 1 more epoch | 0.895 | 0.943 |

Then, the actual training yielded:

| Hyperparameters and Information | Score | Speed |
|-|-|-|
| 4500 samples for 2 epochs      | 0.882 | 0.939 |
| 4500 samples for 6 epochs      | 0.911 | 0.941 |
| 4500 samples for 7 epochs      | 0.877 | 0.941 |
| 4500 samples for 8 epochs      | 0.899 | 0.941 |
| 4500 samples for 9 epochs      | 0.924 | 0.941 |
| 4500 samples for 10 epochs     | 0.949 | 0.940 |
| 4500 samples for 11 epochs     | 0.938 | 0.936 |
| 4500 samples for 12 epochs     | 0.952 | 0.939 |
| 4500 samples for 14 epochs     | 0.932 | 0.941 |

In terms of a graph for the score against epochs, it looks like

![score_against_epochs](/docs/asr/score_against_epochs.png)

COmparing this graph against [team 2B3Y's tuning](https://github.com/ThePyProgrammer/til-25-2b3y/tree/main/asr) yielding 0.98 on the hidden set, it is likely that 12 epochs was still underfitted, but no attempt was made to further improve the score due to time constraints.

### Hyperparameters

Hyperparameters:

* Learning rate: 1e-5
* Warmup: 500 steps
* Epochs: 12
* Optimizer: Adam
* Scheduler: Cosine Annealing
* min_lr: 1e-5
* warmup_steps: 500
* adam_beta1: 0.9
* adam_beta2: 0.999
* weight_decay: 1e-5

A train / validation split of 0.9 / 0.1 was used.

### Inference

The only optimization done was to conduct batch inference in batches of 4 (since it was likely that the endpoint was called with a batch of 4 in qualifiers).

### Additional Steps to Improve

* Batching tensors instead of dumping temporary `.wav` files could have been used, however, in the limited time to do ASR experiments, we could not get batch inference on tensors to work, but [team 2B3Y's inference manager](https://github.com/ThePyProgrammer/til-25-2b3y/tree/main/asr) shows how it can be done.
* Spell checking with standard models or sanity checking with a LLM were considered, however, it was deemed that they would decrease the speed score too much, and were therefore not used. 
* It might be interesting to attempt to train it on a denoised dataset to see the results, or implement k-fold cross validation to improve robustness.

## Hardware Used

Aside from the GCP instance, we also made use of Kaggle's free 2xT4 and TPUv4-8 pod for training.

RL was also trained / evaluated on edge devices (varying architectures )on all members simultaneously to speed things up.

## Final words
We like to thank [Ryan](https://github.com/ryan-tribex), [Ada](https://github.com/HoWingYip) and [Tianshi](https://github.com/qitianshi) for their hard work in setting up the technical side of the competition (which is an impressive combination of various frameworks - ROS, websocketing, networking, hardware infrastructure, etc) and support during the competition.

We would also like to thank DSTA for organizing the largest yearly ML competition in Singapore. Despite not coming in top 4 (largely due to a format which relies heavily on both opponents playing in good faith, and luck), we still believe that the work we have done for this iteration is highly technically interesting, and will prove valuable to any readers.

We would also like to thank team [12000SGDPLUSHIE](https://github.com/aliencaocao/TIL-2024) for a nice format of their previous writeup and training code, which proved to be useful in initial exploratory training.