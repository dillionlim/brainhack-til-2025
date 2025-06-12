# DSTA BrainHack TIL-AI 2025 - Team DeepSheet

<!-- TOC -->
* [Introduction](#introduction)
* [Team Members (in alphabetical order)](#team-members-in-alphabetical-order)
* [Achievements](#achievements)
* [Final evaluation results](#final-evaluation-results)
* [ASR](#asr)
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

[^2]: This was achieved via an unconventional technique, which will be elaborated on, below, hence, the semi-final model would be significantly different.
[^3]: This was achieved using hardware-specific TensorRT optimizations. However, since semi-final participant servers were built on a 5070Ti architecture, which we do not have access to, we ended up switching to a Pytorch implementation with a slightly lower, but comparable speed score.
[^4]: C++ and Python implementations ended up having the exact same accuracy / speed combination, to 3 significant figures.

## Hardware Used

Aside from the GCP instance, we also made use of Kaggle's free 2xT4 and TPUv4-8 pod for training.

RL was also trained / evaluated on edge devices (varying architectures )on all members simultaneously to speed things up.

## Final words
We like to thank [Ryan](https://github.com/ryan-tribex), [Ada](https://github.com/HoWingYip) and [Tianshi](https://github.com/qitianshi) for their hard work in setting up the technical side of the competition (which is an impressive combination of various frameworks - ROS, websocketing, networking, hardware infrastructure, etc) and support during the competition.

We would also like to thank DSTA for organizing the largest yearly ML competition in Singapore. Despite not coming in top 4 (largely due to a format which relies heavily on both opponents playing in good faith, and luck), we still believe that the work we have done for this iteration is highly technically interesting, and will prove valuable to any readers.

We would also like to thank team [12000SGDPLUSHIE](https://github.com/aliencaocao/TIL-2024) for a nice format of their previous writeup and training code, which proved to be useful in initial exploratory training.