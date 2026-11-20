# MRI physics, step by step

A 3D, three-screen Python tutorial (34 steps, 14 phases): scanner anatomy, B0 and net magnetization, RF pulse, T1/T2, slice selection, frequency and phase encoding, pixel to voxel, contrast, pulse sequences, diffusion, k-space and reconstruction.

## Run

    pip install pyvista pyvistaqt PyQt5 matplotlib pillow
    python mri_tutorial.py            # pauses after each phase
    python mri_tutorial.py --auto     # no pauses

Keys: Space pause/play, Right/Left step, Shift+Right/Left phase, 1-4 camera views, mouse to rotate and zoom.

`MRI_Physics_Lesson_Summaries.docx` holds the written lesson summaries (Lesson 1 so far).
