import React, { useState } from "react";
import axios from "axios";

function Video() {
    const [video, setVideo] = useState(null);

    const getVideo = (event) => {
        const file = event.target.files[0]; // Get the first selected file for now
        if (file) {
            setVideo(URL.createObjectURL(file)); // Create a URL for the video file
        }

        // sendVideoToProxy(video);

    };

    const sendVideoToProxy = async (video) => {
        const res = await axios.post(`http://localhost:8000/api/upload_video`, null, {
            params: {
                result: false, //set to false as its not processed yet  
                timestamp: "",
                video: video,
            }
        });
        console.log(res);
    }

    return (
        <div>
            <h1>This is the video component</h1>
            <input
                type="file"
                accept="video/*" //video fule 
                onChange={getVideo} // 
                required
            />
            {video && (
                <video width="600" controls>
                    <source src={video} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
            )}
        </div>
    );
}

export { Video };
