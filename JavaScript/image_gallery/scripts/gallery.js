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


function insertAfter(newElement, targetElement) {
    var parent = targetElement.parentNode;
    if (parent.lastChild == targetElement) {
        parent.appendChild(newElement);
    } else {
        parent.insertBefore(newElement, targetElement.nextSibling);
    }
}

function preparePlaceholder() {
    var placeholder = document.createElement("img");
    placeholder.setAttribute("id", "placeholder");
    placeholder.setAttribute("src", "");
    placeholder.setAttribute("height", "300");
    placeholder.setAttribute("width", "400");
    var description = document.createElement("p");
    description.setAttribute("id", "caption");
    var text = document.createTextNode("Choose an image");
    description.appendChild(text);
    var gallery = document.getElementById("imagegallery");
    insertAfter(placeholder, gallery);
    insertAfter(description, placeholder);
    return true;
}

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

addLoadEvent(prepareLinks);
addLoadEvent(preparePlaceholder);
