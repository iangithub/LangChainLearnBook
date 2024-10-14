let audioRecorder;
let audioChunks = [];
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        audioRecorder = new MediaRecorder(stream);
        // dataavailable event is triggered when the recording is finished
        audioRecorder.addEventListener('dataavailable', e => {
            audioChunks.push(e.data);
            console.log('Data available');
            uploadVoice();
        });
        $("#start").click(function () {
            if (audioRecorder.state == 'inactive') {
                $("#start").text("Stop");
                audioChunks = [];
                audioRecorder.start();
                console.log(audioRecorder.state);
            } else {
                audioRecorder.stop();
                console.log(audioRecorder.state);
                $("#start").text("Record");
            }
        });
    }).catch(err => {
        // If the user denies permission to record audio, then display an error.
        console.log('Error: ' + err);
    });

function uploadVoice() {
    const blobObj = new Blob(audioChunks, { type: 'audio/webm' });
    const audioFile = new File([blobObj], 'audio.webm', { type: 'audio/webm' });
    // send audio file to server
    const formData = new FormData();
    formData.append('audio', audioFile);
    fetch('/upload', {
        method: 'POST',
        body: formData
    }).then(response => {
        console.log(response);
        return response.json();
    }).then(data => {
        console.log(data);
        console.log(data["whisper"]);
        console.log(data["openai"]);
        console.log(data["voice"]);
        writeOut(data["whisper"], data["openai"]);
        speakOut(data["voice"]);
    }).catch(err => {
        console.log('Error: ' + err);
    });
}

function writeOut(whisper, openai) {
    $("#dialog").append("我 : " + whisper + "\n");
    $("#dialog").append("AI : " + openai + "\n");
    //make textarea scroll to the bottom
    $("#dialog").scrollTop($("#dialog")[0].scrollHeight);
}

function speakOut(voice_url){
    const audio_tts = new Audio(voice_url + "?a=" + Math.random());
    audio_tts.play();
}

$(function () {
    $("#clearHistory").click(clearHistory);
    $(window).on("beforeunload", clearHistory);
    $("#message").keypress(function (e) {
        if (e.which == 13) {
            chatWithLLM();
        }
    });
});

function chatWithLLM() {
    var message = $("#message").val();
    $("#dialog").append("我 : " + message + "\n");
    var data = {
        message: message
    };
    $.post("/call_llm", data, function (data) {
        $("#dialog").append("AI : " + data + "\n");
        //make textarea scroll to the bottom
        $("#dialog").scrollTop($("#dialog")[0].scrollHeight);
    });
    $("#message").val("");
    //make textarea scroll to the bottom
    $("#dialog").scrollTop($("#dialog")[0].scrollHeight);
}


function clearHistory() {
    $("#dialog").empty();
    $.post("/clear_history", {}, function (data) {
        console.log(data);
    });
}
