document.querySelector('.sendButton').addEventListener('click', function() {
    const newsText = document.getElementById("newsText").value;
    const responseDisplay = document.getElementById("responseDisplay");

    if (newsText) {
        chrome.runtime.sendMessage({action: "checkFakeNews", text: newsText}, function(response) {
            responseDisplay.style.display = 'block'; // Make the response box visible
            if (response && response.result) {
                // Ensure that 'result' field is present in the response
                responseDisplay.textContent = response.result;
            } else {
                // Handle the case where 'result' might be missing or undefined
                responseDisplay.textContent = "Error or no response received";
            }
        });
    } else {
        // Handle the case where no text is entered
        responseDisplay.textContent = "Please enter some text to check.";
    }
});
