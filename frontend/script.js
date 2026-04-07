document.getElementById("analyzeBtn").addEventListener("click", async () => {

    const text = document.getElementById("textInput").value.trim();

    if (!text) {
        alert("Please enter some text!");
        return;
    }

    // API ROUTE
    const API_URL = "http://127.0.0.1:8000/analyze"; 

    const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    // Extract results
    let toxicLabel = data.toxicity.label;
    let toxicConf = data.toxicity.confidence;
    let sentiLabel = data.sentiment.label;
    let sentiConf = data.sentiment.confidence;

    // **Post-processing rule**
    // If toxicity is offensive or hate, override sentiment → NEGATIVE
    if (toxicLabel !== "safe") {
        sentiLabel = "negative";
    }

    // Update UI
    document.getElementById("resultBox").classList.remove("hidden");

    document.getElementById("sentimentLabel").innerText = sentiLabel.toUpperCase();
    document.getElementById("sentimentConf").innerText = `confidence: ${sentiConf}`;

    document.getElementById("toxicityLabel").innerText = toxicLabel.toUpperCase();
    document.getElementById("toxicityConf").innerText = `confidence: ${toxicConf}`;
});
