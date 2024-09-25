import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from src.dal import DetectionDAL
from bson import ObjectId

# ENVS/MAJOR VARS need to move
COLLECTION_NAME = "detection_results"
MONGODB_URL = os.environ["MONGODB_URL"]
print(MONGODB_URL)
DEBUG = os.environ.get("DEBUG", "").strip().lower() in {
    "1", "true", "on", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI Lifespan manager for MongoDB connection
    # Startup:
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client["default"]

    # Ensure the database is available:
    try:
        pong = await database.command("ping")
        if int(pong["ok"]) != 1:
            raise Exception("Cluster connection is not okay!")
    except Exception as e:
        raise Exception(f"Error connecting to MongoDB: {str(e)}")

    # Get the detection results collection
    detection_results = database[COLLECTION_NAME]

    # Print all documents in the collection
    try:
        async for document in detection_results.find():
            print(document)
    except Exception as e:
        print(f"Error fetching documents: {str(e)}")

    # Attach the DAL to the app instance
    app.state.detection_dal = DetectionDAL(detection_results)

    # Yield control back to the FastAPI application
    yield  # This is where control is yielded back

    # Shutdown:
    client.close()

# Create FastAPI app with MongoDB connection lifespan and debug mode
app = FastAPI(lifespan=lifespan, debug=DEBUG)


@app.post("/api/results/{video_id}/detection")
async def store_detection_result(video_id: str, result: str, timestamp: str = None):
    print(
        f"Received: video_id={video_id}, result={result}, timestamp={timestamp}")
    try:
        print("Im am here post")
        # Call the async DAL function to add a new detection result
        upload_vid = await app.state.detection_dal.add_detection_result(
            video_id, result, timestamp
        )
        if upload_vid:
            return {"status": "success", "submitted": upload_vid}
        else:
            raise HTTPException(
                status_code=404, detail="No video")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error storing video: {str(e)}")


@app.get("/api/results/{video_id}")
async def get_video_analysis(video_id: str):
    print("Im am here GET")
    try:
        # Call the async DAL function to retrieve the video analysis
        result = await app.state.detection_dal.get_video_analysis(video_id)
        print(result)
        # Check if a result was found for the given video_id
        if result:
            return result
        else:
            raise HTTPException(
                status_code=404, detail="Video analysis not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving video analysis: {str(e)}")


# Entry point for running the app
def main(argv=sys.argv[1:]):
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=3001, reload=DEBUG)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
