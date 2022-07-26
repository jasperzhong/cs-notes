# JavaScript

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

**内建对象**: 比如`Array`, `Math`, `Date`

```js
var num = Math.round(3.14);
var current_date = new Date();
```

**宿主对象**: 这些对象不是由JavaScriptd语言本身而是由它的运行环境提供的, 比如`document`对象. 


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

