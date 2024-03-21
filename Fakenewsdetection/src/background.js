chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === "checkFakeNews") {
        const text = request.text;

        // Send 'text' to Flask server and receive response
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        })
        .then(response => response.json())
        .then(data => {
            // Handle the response from the Flask server
            // The response format is expected to match the Flask server's output
            sendResponse({ result: data.prediction });
        })
        .catch(error => {
            console.error('Error:', error);
            sendResponse({ result: 'Error in processing the request' });
        });

        return true; // Indicates that the response is asynchronous
    }
});
