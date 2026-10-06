import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.camera.camera_stream import (
    generate_frames,
    start_camera,
    start_camera_from_file,
    stop_camera,
    pause_camera,
    resume_camera,
    set_zone,
    set_zone_enabled,
)

router = APIRouter()

# Jahan upload ki hui videos save hongi - project ke andar hi
# (video.py: app/api/ -> 3 level upar project root, phir data/uploaded_videos)
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "uploaded_videos"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ZoneUpdate(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ZoneToggle(BaseModel):
    enabled: bool


class VideoFileStart(BaseModel):
    path: str          # server ke disk par file ka poora ya relative path
    loop: bool = True   # khatam hote hi khud dobara shuru ho


@router.get("/video-feed")
def video_feed():
    return StreamingResponse(
        generate_frames(cam_id=1),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/video-feed/{cam_id}")
def video_feed_by_id(cam_id: int):
    return StreamingResponse(
        generate_frames(cam_id=cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/camera/start")
def camera_start():
    success = start_camera(cam_id=1)
    return {"status": "started" if success else "failed"}


@router.post("/camera/start/{cam_id}")
def camera_start_by_id(cam_id: int, device_index: int = 0):
    success = start_camera(cam_id=cam_id, device_index=device_index)
    return {"status": "started" if success else "failed"}


@router.post("/camera/start-file/{cam_id}")
def camera_start_from_file(cam_id: int, body: VideoFileStart):
    """
    Live webcam ki jagah ek recorded .mp4 (jo server ke disk par pehle se
    maujood ho, jaise data/test_videos/theft_clip.mp4) ko "camera" ki tarah
    chalata hai. Path file ke poore ya project-root se relative address ke
    tor par de sakte hain. Dashboard baaki sab kuch (Start/Pause/Stop,
    video-feed) waisay hi istemal karega jaisa live camera ke liye karta hai.
    """
    success = start_camera_from_file(cam_id=cam_id, file_path=body.path, loop=body.loop)
    return {"status": "started" if success else "failed (file nahi khuli — path check karein)"}


@router.post("/camera/upload-video/{cam_id}")
async def camera_upload_video(cam_id: int, file: UploadFile = File(...), loop: bool = True):
    """
    Dashboard se browser ke file-picker ke zariye choose ki hui video
    yahan upload hoti hai, server par (data/uploaded_videos/) save hoti
    hai, aur turant usi camera slot par "chal" bhi jati hai - is se
    dashboard par ek dropdown option "Video File" bana kar seedha yahan
    POST kiya ja sakta hai.
    """
    safe_name = f"cam{cam_id}_{file.filename}"
    dest_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(dest_path, "wb") as f:
        while chunk := await file.read(2080 * 2080):
            f.write(chunk)

    success = start_camera_from_file(cam_id=cam_id, file_path=dest_path, loop=loop)
    return {
        "status": "started" if success else "failed (file save hui lekin khul nahi payi)",
        "saved_path": dest_path,
    }


@router.post("/camera/pause")
def camera_pause():
    pause_camera(cam_id=1)
    return {"status": "paused"}


@router.post("/camera/pause/{cam_id}")
def camera_pause_by_id(cam_id: int):
    pause_camera(cam_id=cam_id)
    return {"status": "paused"}


@router.post("/camera/resume")
def camera_resume():
    resume_camera(cam_id=1)
    return {"status": "resumed"}


@router.post("/camera/resume/{cam_id}")
def camera_resume_by_id(cam_id: int):
    resume_camera(cam_id=cam_id)
    return {"status": "resumed"}


@router.post("/camera/stop")
def camera_stop():
    stop_camera(cam_id=1)
    return {"status": "stopped"}


@router.post("/camera/stop/{cam_id}")
def camera_stop_by_id(cam_id: int):
    stop_camera(cam_id=cam_id)
    return {"status": "stopped"}


@router.post("/camera/zone/{cam_id}")
def camera_zone_update(cam_id: int, zone: ZoneUpdate):
    """Frontend se naye zone coordinates (0.0 - 1.0 ratios) receive karke save karta hai."""
    set_zone(cam_id, zone.x1, zone.y1, zone.x2, zone.y2)
    return {"status": "zone_updated"}


@router.post("/camera/zone/{cam_id}/toggle")
def camera_zone_toggle(cam_id: int, toggle: ZoneToggle):
    """Zone ko is camera ke liye on/off karta hai."""
    set_zone_enabled(cam_id, toggle.enabled)
    return {"status": "zone_enabled" if toggle.enabled else "zone_disabled"}