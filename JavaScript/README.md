# JavaScript

## JavaScript语法

语法上和Python基本差不多. 总结几个要点.

1. JS不需要进行类型声明，是一种弱类型(weakly typed)语言. 

基本数据类型和Python一样.
- 字符串. 
    - 单引号双引号都可以. 
    - 字符串可以用"+"拼接，和Python一样. 甚至可以直接str + number，而不需要转换. 
- 数值. 

2. 数组

数组和Python中的list差不多, 可以放任何东西. 

创建数组
```js
// 不给出元素个数
var beatles = Array();

// 指定元素个数
var beatles = Array(4);

// 同时初始化
var beatles = Array("John", "Paul", "George", "Ringo");

// 直接用中括号
var beatles = ["John", "Paul", "George", "Ringo"];
```

关联数组 

```py
var lennon = Array();
lennon["name"] = "John";
lennon["year"] = 1940;
lennon["living"] = false;
```

不推荐使用，因为这个也可以用对象来存. 


4. 对象

使用`Object`关键词. 这个确实比Python方便. 

`Object.property`, `Object.method()`

```js
var lennon = Object();
lennon.name = "John";
lennon.year = 1940;
lennon.living = false;

// 或者用花括号语法
var lennon = {name: "John", year: 1940, living: false}
```

花括号语法很像Python的dict. 

三种类型对象: 
- 内建对象: 比如`Array`, `Math`, `Date`

```js
var num = Math.round(3.14);
var current_date = new Date();
```

- **宿主对象**: 这些对象不是由JavaScriptd语言本身而是由它的运行环境提供的, 比如`document`对象. 

- 用户自定义对象.



5. 语句

分支和C/C++差不多. if, else if, else. 主要注意`==`和`===`区别. 后者不仅比较值，还会比较类型. 

```js
var a = false;
var b = "";
if (a == b) {
    alert("a equals b");
}
```

这个条件语句的求值结果为true, 因为`==`认为空字符串和false的含义相同. 

对于`!=`和`!==`同理. 


循环和C/C++差不多. while, do while, for. 

```js
var beatles = ["John", "Paul", "George", "Ringo"];
for (var i = 0; i < beatles.length; i++) {
    alert(beatles[i]);
}
```

也可以用类似于Python的for
```js
for (var name in beatles) {
    alert(name);
}

```

6. 函数

注意`var`关键词可以明确为函数变量作用域.
- 如果在某个函数中使用了`var`，那个变量就会被视为一个局部变量. 
- 如果没有使用`var`, 那个变量就将被是为一个全局变量，如果脚本里已经存在与之同名的全局变量，这个函数就会改变那个全局变量的值. 


```js
function squre(num) {
    total = num * num; // this changes var total below
    return total;
}

var total = 50;
var number = square(20);
alert(total);
```

## DOM

Dcoument Object Model (DOM)

DOM把一份文档表示为tree. 用parent, child, sibling表示之间关系. 

节点，每个节点有一个`nodeType`属性. 
- **元素节点(element node)**. 比如文本段落元素"p", 无序清单元素"ul", 列表项元素"li". `nodeType=1`.
- 文本节点(text node). 例如<p>元素包含文本"hello world". `nodeType=2`.
- 属性节点(attribute node). 例如`<p title='test'>hello world</p>`, "title='test'"就是属性节点. `nodeType=3`.



### CSS 

层叠样式表. 对样式的声明放在<head>部分的<style>标签之间. 

CSS声明元素样式与JavaScript函数定义语法相似:
```css
selector {
    property: value;
}
```

CSS的一个特点是**继承**, 即DOM上各个元素继承其父元素的样式属性. 

例如
```css
body {
    color: white;
    background-color: black;
}
```

这些颜色不仅作用于body，而且作用于嵌套在body元素内部的所有元素. 

为了作用于特定元素，需要使用class属性后者id属性. 

**class属性**

可以在所有元素上任意应用class属性.
```html
<p class="special">hello world</p>
<h2 class="speial">hello world</p>
```

在样式表中, 为class属性相同的所有元素定义同一种样式:
```css
.special {
    font-style: italic;
}
```

还可以为一种特定类型的元素定义一种特定的样式:
```css
h2.special {
    text-transform: uppercase;
}
```

**id属性**

id属性的用途是给网页里某个元素加上一个独一无二的标识符. 
```html
<ul id='purchases'>
```

在样式表中，可以为有特定id属性值的元素定义一种独享的样式:
```css
#purchases {
    border: 1px solid white;
    background-color: #333;
    color: #ccc;
    padding: 1em;
}
```

尽管id本身只能使用一次, 样式表还是可以利用id属性为包含在该特定元素里的其他元素定义样式.
```css
#purchases li {
    font-weight: bold;
}
```

id属性像一个挂钩，一头连着文档里某个元素，另一头连着CSS样式表里的某个样式. 

### 获取元素

如何获取元素. document每个元素节点都是Object. 下面三个操作返回的都是Object(s). 

1. `document.getElementById(id)`: 返回一个Object, 其id属性值为传入的id. 
2. `document.getElementsByTagName(tag)`: 返回具有相同标签的Object数组. tag就是标签的名字，比如`li`. 甚至可以用通配符"*".
3. `document.getElementsByClassName(class)`: 返回具有相同类名的元素的Object数组.


### 获取和设置属性

前面三个方法是如何获取元素. 获取元素后，可以获取它的各个属性. 

1. `object.getAttribute(attribute)`

```js
var paras = document.getElementsByTagName("p");
for (var i = 0; i < paras.length; i++) {
    alert(paras[i].getAttribute("title"));
}
```

2. `object.setAttribute(attribute, value)`

如果`attribute`不存在，则创建一个属性. 若存在则覆盖.

setAttribute对文档做出修改后是立刻生效的. 但在浏览器view source查看文档源代码却还是改变前的属性值. 也就是说，setAttribute做出的修改不会反映在文档本身的源代码里面. 

DOM的工作模式: 先加载文档的静态内容，再动态刷新，动态b刷新不影响文档的静态内容. 所谓动态刷新，即对页面内容进行刷新却不需要再浏览器里刷新页面.


一些属性:
- childNodes
- nodeType
- nodeValue
- firstChild
- lastChild

### 创建元素

目的是动态修改网页结构. 

- `document.createElement(nodeName)`: 创建一个元素节点. 
- `parent.appendChild(child)`: 插入到parent节点作为子节点
- `document.createTextNode(text)`: 创建一个文本节点


```js
window.onload = funcion() {
    var para = document.createElement("p");
    var testdiv = document.getElementById("testdiv");
    textdiv.appendChild(para);
    var txt = document.createTextNode("Hello world!");
    para.appendChild(txt);
}
```

- `parentElement.insertBefore(newElement, targetElement)`

但是DOM居然没有`insertAfter`方法...但可以自己实现一个
```js
function insertAfter(newElement, targetElement) {
    var parent = targetElement.paraentNode;
    if (parent.lastChild == targetElement) {
        parent.appendChild(newElement);
    } else {
        parent.insertBefore(newElement, targetElement.nextSibling);
    }
}
```



- 结构(html)、行为(javascript)和样式(css)分离

### Ajax

异步请求数据. 关键的类是`XMLHttpRequest`，其中最有用的方法是`open`，它用来指定服务器上将要访问的文件，指定请求类型: GET, POST或者SEND. 

```js
function getNewContent() {
    var request = new XMLHttpRequest();
    request.open("GET", "example.txt", true);
    request.onreadystatechange = function() {
        if (request.readState == 4) {
            var para = document.createElement("p");
            var txt = document.createTextNode(request.responseText);
            para.appendChild(txt);
            document.getElementById('new').appendChild(para);
        }
    }
    request.send(null);
}
```

注意Ajax的**异步性**: 脚本在发送XMLHttpRequest请求后，仍会继续执行，不会等待响应返回. 

代码中的`onreadstatechange`是一个事件处理函数，它会在服务器给XMLHttpRequst对象发送回响应的时候被触发. 

```js
request.onreadystatechange = doSomething;
```

`readState`属性
- 0: 未初始化
- 1: 正在加载
- 2: 加载完毕
- 3: 正在交互
- 4: 完成

访问服务器发回来的数据通过: 
- `responseText`属性
- `responseXML`属性


