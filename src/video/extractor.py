import logging
import cv2 as cv

logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str) -> tuple[float, int, int, int]:
    """
    Read video's metadata from the source path:
    Output: Video's: FPS, Size (width x height), Number of frames
    """
    logger.debug(f"Opening video file to extract metadata: {video_path}")
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise FileNotFoundError(f"Can not open a video file: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()

    logger.debug(
        f"Video metadata extracted: {width}x{height}, FPS={fps:.2f}, total_frames={frame_count}"
    )
    return fps, width, height, frame_count


def chunk_video(
    chunk_t: int, chunk_extr_frames: int, overlap_t: int, fps: float, frame_count: int
) -> list[list[int]]:
    """
    Calculates frames indices to extract based on the input values:
    Input: Chunk time, Frames per chunk, Overlap between chunks, FPS and Frame count
    Output: List of frame indices to extract
    """
    logger.debug(
        f"Chunking video: chunk_t={chunk_t}s, overlap_t={overlap_t}s, target_frames_per_chunk={chunk_extr_frames}"
    )
    chunk_total_frames = int(chunk_t * fps)
    overlap_frames = int(overlap_t * fps)
    stride_frames = chunk_total_frames - overlap_frames
    if stride_frames <= 0:
        logger.error(
            f"Invalid overlap configuration: overlap_t ({overlap_t}s) must be shorter than chunk_t ({chunk_t}s)"
        )
        raise ValueError("Overlap time has to be shorter than chunk time.")

    all_chunks_indices = []

    for start_frame in range(0, frame_count, stride_frames):
        end_frame = min(start_frame + chunk_total_frames, frame_count)
        current_chunk_length = end_frame - start_frame
        current_chunk_indices = [
            int(
                start_frame
                + ((current_chunk_length - 1) * i / max(1, chunk_extr_frames - 1))
            )
            for i in range(chunk_extr_frames)
        ]

        current_chunk_indices = sorted(list(set(current_chunk_indices)))
        all_chunks_indices.append(current_chunk_indices)

        if end_frame == frame_count:
            break

    logger.debug(f"Partitioned video into {len(all_chunks_indices)} chunk(s)")
    return all_chunks_indices


def extract_frames(video_path: str, all_chunks_indices: list):
    """
    Extracting frames from video based on the metadata.
    Input: Video path & list of frame indices
    Output: List of extracted frames converted to RGB format
    """
    logger.debug(
        f"Initializing frame extraction from {video_path} for {len(all_chunks_indices)} chunk(s)"
    )
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file for extraction: {video_path}")
        raise FileNotFoundError(f"Can not open a video file: {video_path}")

    try:
        for chunk_idx, chunk in enumerate(all_chunks_indices):
            frames = []
            for frame_idx in chunk:
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                else:
                    logger.warning(
                        f"Failed to read frame at index {frame_idx} (chunk {chunk_idx + 1})"
                    )

            if frames:
                logger.debug(
                    f"Yielding batch of {len(frames)} frames for chunk {chunk_idx + 1}"
                )
                yield frames

    finally:
        cap.release()
        logger.debug(f"Closed video capture stream for {video_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    VIDEO_PATH = "data/input/videoplayback.mp4"
    TIME_OF_CHUNK = 10
    EXTRACTED_FRAMES_PER_CHUNK = 4
    OVERLAP_T = 2

    fps, width, height, frame_count = get_video_metadata(VIDEO_PATH)
    all_idxs = chunk_video(
        TIME_OF_CHUNK, EXTRACTED_FRAMES_PER_CHUNK, OVERLAP_T, fps, frame_count
    )
    video_chunks_generator = extract_frames(VIDEO_PATH, all_idxs)

    for idx, frames_batch in enumerate(video_chunks_generator):
        logger.info(f"Chunk {idx + 1}: Frames per batch: {len(frames_batch)}")
