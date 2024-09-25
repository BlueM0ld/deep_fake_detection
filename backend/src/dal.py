from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

from pymongo import ReturnDocument

# Model for a single detection result


class DetectionResult(BaseModel):
    id: str
    video_id: str
    result: str  # Could fake(True) or real(false)
    timestamp: str  # Time when the detection result was generated

    @staticmethod
    def from_doc(doc) -> "DetectionResult":
        return DetectionResult(
            id=str(doc["_id"]),
            video_id=doc["video_id"],
            result=doc["result"],
            timestamp=doc["timestamp"],
        )


# Model for a video file with detection results
# We will be looking frame by frame!

class VideoAnalysis(BaseModel):
    id: str
    video_id: str
    detection_results: List[DetectionResult]

    @staticmethod
    def from_doc(doc) -> "VideoAnalysis":
        return VideoAnalysis(
            id=str(doc["_id"]),
            video_id=doc["video_id"],
            detection_results=[
                DetectionResult.from_doc(result) for result in doc["detection_results"]
            ],
        )

# Data Access Layer for managing detection results


class DetectionDAL:
    def __init__(self, detection_collection: AsyncIOMotorCollection):
        self._detection_collection = detection_collection

    # List all video analysis records
    async def list_video_analyses(self, session=None):
        async for doc in self._detection_collection.find({}, session=session):
            yield VideoAnalysis.from_doc(doc)

    # Create a new video analysis record
    async def create_video_analysis(self, video_id: str, session=None) -> str:
        response = await self._detection_collection.insert_one(
            {"video_id": video_id, "detection_results": []},
            session=session,
        )
        return str(response.inserted_id)

    # Get a specific video analysis record by video_id
    async def get_video_analysis(self, video_id: str | ObjectId, session=None) -> VideoAnalysis:
        doc = await self._detection_collection.find_one({"video_id": video_id})
        if doc:
            vid_doc = DetectionResult.from_doc(doc)
            return vid_doc

    # Delete a video analysis record by video_id

    async def delete_video_analysis(self, video_id: str | ObjectId, session=None) -> bool:
        response = await self._detection_collection.delete_one(
            {"video_id": video_id},
            session=session,
        )
        return response.deleted_count == 1

    # Add a new detection result to an existing video analysis

    async def add_detection_result(
        self,
        video_id: str | ObjectId,
        result: str,
        timestamp: str | None,
        session=None,
    ) -> VideoAnalysis | None:
        detection_result = {
            "video_id": video_id,
            "result": result,
            "timestamp": timestamp,
        }

        insert_result = await self._detection_collection.insert_one(
            detection_result
        )

        if insert_result:
            return "True"
