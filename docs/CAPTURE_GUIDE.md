# How to film the two videos

The system is built to cope with different viewpoints, so you don't need to repeat the same path. A few habits make the 3D reconstruction much better, though.

## Both videos
- **Portrait or landscape**, the **same orientation** for both videos.
- **20–40 seconds** each, walking **slowly** around one side of the room. Sideways steps are better than rotating on the spot: the 3D comes from parallax.
- Hold the phone at **chest height**, tilted slightly down so the **floor is visible** in most frames. The floor fixes gravity and the metric scale. If you film from a different height, pass `--camera-height`.
- Keep the lights the same. Avoid filming straight into a window or a mirror.
- Nobody walking through the shot.

## Between the videos: make 3–5 changes
A good mix that shows everything the system does:
- **Move** one piece of furniture by at least ~0.5 m, and turn it (a chair is ideal).
- **Remove** something small from a surface (e.g. a plant or a bottle from a table).
- **Add** an obstacle on the floor (a backpack or a box).
- Optional: add something unusual that isn't in the detector's list. It should show up as an "unidentified object".
- Optional: take something flat off the floor or a table (a doormat, a rug, some papers). It has no height, so it should show up as "looks different" from colour alone.
- Optional: leave something in a corner that the second video *won't* film. It should come back as "can't tell" rather than "removed".

## The second video
- Start from a **different spot** and walk a **different path**. About half of what the first video saw should be visible again, from new angles.
- Make sure every place where you changed something is in view at some point: old positions, new positions, and the empty spots.

## Then
```bash
python -m roomdiff run --before before.mp4 --after after.mp4 --out runs/my_room
```
or use the Colab notebook. If the diagnostics report a low overlap, film the second video with more of the room in common.
