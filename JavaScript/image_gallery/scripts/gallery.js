function addLoadEvent(func) {
    var oldOnLoad = window.onload;
    if (typeof window.onload != 'function') {
        window.onload = func;
    } else {
        window.onload = function() {
            oldOnLoad();
            func();
        }
    }
}

function prepareLinks() {
    var links = document.getElementById("imagegallery").getElementsByTagName("a");
    for (var i = 0; i < links.length; i++) {
        if (links[i].tagName == "A") {
            links[i].onclick = function () {
                return !showPic(this);
            }
        }
    }
}

addLoadEvent(prepareLinks);

function showPic(whichpic) {
    var source = whichpic.getAttribute("href");
    var placeholder = document.getElementById("placeholder");
    if (!placeholder) {
        return false;
    }
    placeholder.setAttribute("src", source);

    var text = whichpic.firstChild.nodeValue;
    var caption = document.getElementById("caption");
    if (!caption) {
        return false;
    }
    caption.firstChild.nodeValue = text;
    return true;
}
