import React, { useState } from "react";
import axios from "axios";

function App() {
  const [videoId, setVideoId] = useState(""); // State for the video ID input
  const [result, setResult] = useState(""); // State for the result input
  const [response, setResponse] = useState(null); // State for the API response
  const [fetchVideoId, setFetchVideoId] = useState(""); // State for fetching a result
  const [fetchedResult, setFetchedResult] = useState(null); // State to display fetched results

  // Handle form submission for adding detection results
  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log("Print here")
    try {
      const res = await axios.post(`http://localhost:8000/api/results/${videoId}/detection`, null, {
        params: {
          result: result,  // Pass result in the query string
          timestamp: "" // Pass timestamp in the query string (optional)
        }
      });

      setResponse(res.data);
      console.log("Response:", res.data);

    } catch (error) {
      console.error("Error:", error);
      setResponse("Error storing detection result.");
    }
  };

  // Handle form submission for fetching detection results by video ID
  const handleFetchResult = async (e) => {
    e.preventDefault();

    try {
      const res = await axios.get(`http://localhost:8000/api/results/${fetchVideoId}`);
      setFetchedResult(res.data);
      console.log("Fetched Result:", res.data);
    } catch (error) {
      console.error("Error fetching result:", error);
      setFetchedResult("Error fetching detection result.");
    }
  };

  return (
    <div className="App">
      <h1>Deepfake Detection Results</h1>

      {/* Form for submitting detection results */}
      <form onSubmit={handleSubmit}>
        <h2>Add Detection Result</h2>
        <label>
          Video ID:
          <input
            type="text"
            value={videoId}
            onChange={(e) => setVideoId(e.target.value)}
            required
          />
        </label>
        <br />
        <label>
          Result:
          <input
            type="text"
            value={result}
            onChange={(e) => setResult(e.target.value)}
            required
          />
        </label>
        <br />
        <button type="submit">Submit Detection Result</button>
      </form>

      {/* Display the response for submitting results */}
      {response && (
        <div>
          <h3>Response</h3>
          <p>{JSON.stringify(response)}</p>
        </div>
      )}

      <hr />

      {/* Form for fetching detection results */}
      <form onSubmit={handleFetchResult}>
        <h2>Fetch Detection Result by Video ID</h2>
        <label>
          Video ID:
          <input
            type="text"
            value={fetchVideoId}
            onChange={(e) => setFetchVideoId(e.target.value)}
            required
          />
        </label>
        <br />
        <button type="submit">Fetch Detection Result</button>
      </form>

      {/* Display the fetched result */}
      {fetchedResult && (
        <div>
          <h3>Fetched Detection Result</h3>
          <pre>{JSON.stringify(fetchedResult, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
