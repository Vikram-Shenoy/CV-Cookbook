# CV-Cookbook
A cookbook for Computer Vision Algorithms, notes, and programs.

## Virtual mouse
- Currently using mediapipe, opencv and pyautoGUI to detect hand gesture which will translate to mouse movement, scroll, click. Just by hoevering your hand in the air.


## Vehicle Speed Detection
- Apply various methods to determine speed of vehicles on a highway.
### 1. Distance - Time Estimation
What is it?
- Draw two parrallel lines, a known distance apart.
- Divide the time it takes for a vehicle to cross this known distance, to get the speed of the vehicle.

<p align="center">
<img src="https://github.com/user-attachments/assets/10cec959-ed64-4788-87ad-21b6d60463a3" alt="Example frame from output video" width="650" height="400">
</p>
- The distance between these two lines is determined based on environmental cues. In our case I have used the known distance between the gap in the highway shoulder markings on french highways.

### 2. Optical Flow - Lucas Kanade Method[Work in progress]

<p align="center">

<img src="https://github.com/user-attachments/assets/de508d59-1d07-4f82-ba26-1325c38e2498" alt="Example frame from output video" width="650" height="400">

</p>

## Face Recoginition [Work In Progress]
- Gather Face information and name assigned from a folder, and detect if that face is present in the current frame.
