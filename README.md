### Hi there 👋

This is the repo for The Metamorphosis: Automatic Detection of Scaling Issues for Mobile Apps. It contains the python implementation of dVermin.

dVermin contains three phases, which are view mapping, inter-view inconsistent detection and intra-view detection
analysis. This implementation will analyze the three view trees and the corresponding view image of three scale size
combinations. This implementation is mainly in `dVermin.py` file.

`layout-extractor.jar` is the jar for extracting VT and images form android runtime. Please run your eumulator, and run your app with mapping ids inside, then run this jar with Java of 1.8.
The first parameter is the package name you want to analyze; the second one is the output dir path; The third one is the path to your adb path on your computer; the last one is the path to your Android sdk on computer. Please choose the emulator from AOSP.

You could check out our dataset sample in `https://figshare.com/s/d60421220e6a3fcd27c9`.
