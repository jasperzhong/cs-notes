window.onload = prepareLinks;
function prepareLinks() {
    var links = document.getElementsByClassName("gallery");
    for (var i = 0; i < links.length; i++) {
        if (links[i].className == "gallery") {
            links[i].onclick = function () {
                showPic(this);
                return false;
            }
        }
    }
}

function showPic(whichpic) {
    var source = whichpic.getAttribute("href");
    var placeholder = document.getElementById("placeholder");
    placeholder.setAttribute("src", source);

    var text = whichpic.firstChild.nodeValue;
    var caption = document.getElementById("caption");
    caption.firstChild.nodeValue = text;
}
