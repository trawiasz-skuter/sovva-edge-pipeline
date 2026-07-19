from video.extractor import get_video_metadata, chunk_video, extract_frames
from config import settings


if __name__ == "__main__":
    fps, _, _, frame_count = get_video_metadata(settings.video_path)
    all_idxs = chunk_video(
        settings.chunk_t, settings.chunk_frames, settings.overlap, fps, frame_count
    )
    video_chunks_generator = extract_frames(settings.video_path, all_idxs)
