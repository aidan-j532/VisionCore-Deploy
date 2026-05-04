from VisionCore.config.VisionCoreConfig import VisionCoreConfig
import logging

class ExampleCustomTracker:
    def __init__(self, config: VisionCoreConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.object_count = 0
        self.logger.info("ExampleCustomTracker initialized")

    def process_detections(self, detections):
        self.object_count += len(detections)
        self.logger.info(f"Processed {len(detections)} detections. Total count: {self.object_count}")

        # Example processing: return count statistics
        return {
            'total_count': self.object_count,
            'current_batch': len(detections),
            'average_per_frame': self.object_count / max(1, self.object_count)  # Simplified
        }

    def reset(self):
        self.object_count = 0
        self.logger.info("ExampleCustomTracker reset")

    def get_status(self):
        return {
            'tracker_type': 'example_custom',
            'total_objects_counted': self.object_count
        }