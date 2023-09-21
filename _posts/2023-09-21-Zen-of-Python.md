---
layout: post
title: （转载）Python编程建议
categories: [Tutorial]
description: 90条Python编程建议
keywords: python, tutorial
mermaid: false
sequence: false
flow: false
mathjax: false
mindmap: false
mindmap2: false
---

## **1. 首先**

建议1、理解 Pythonic 概念—-详见 Python 中的《Python之禅》

建议2、编写 Pythonic 代码

（1）避免不规范代码，比如只用大小写区分变量、使用容易混淆的变量名、害怕过长变量名等。有时候长的变量名会使代码更加具有可读性。

（2）深入学习 Python 相关知识，比如语言特性、库特性等，比如Python演变过程等。深入学习一两个业内公认的 Pythonic 的代码库，比如Flask等。

建议3：理解 Python 与 C 的不同之处，比如缩进与 {}，单引号双引号，三元操作符？， Switch-Case 语句等。

建议4：在代码中适当添加注释

建议5：适当添加空行使代码布局更加合理

建议6：编写函数的 4 个原则

（1）函数设计要尽量短小，嵌套层次不宜过深

（2）函数声明应该做到合理、简单、易用

（3）函数参数设计应该考虑向下兼容

（4）一个函数只做一件事，尽量保证函数粒度的一致性

建议7：将常量集中在一个文件，且常量名尽量使用全大写字母

## **2. 编程惯用法**

建议8：利用 assert 语句来发现问题，但要注意，断言 assert 会影响效率

建议9：数据交换值时不推荐使用临时变量，而是直接 a, b = b, a

建议10：充分利用惰性计算（Lazy evaluation）的特性，从而避免不必要的计算

建议11：理解枚举替代实现的缺陷（最新版 Python 中已经加入了枚举特性）

建议12：不推荐使用 type 来进行类型检查，因为有些时候 type 的结果并不一定可靠。如果有需求，建议使用 isinstance 函数来代替

建议13：尽量将变量转化为浮点类型后再做除法（Python3 以后不用考虑）

建议14：警惕eval()函数的安全漏洞，有点类似于 SQL 注入

建议15：使用 enumerate() 同时获取序列迭代的索引和值

建议16：分清 == 和 is 的适用场景，特别是在比较字符串等不可变类型变量时（详见评论）

建议17：尽量使用 Unicode。在 Python2 中编码是很让人头痛的一件事，但 Python3 就不用过多考虑了

建议18：构建合理的包层次来管理 Module

## **3. 基础用法**

建议19：有节制的使用 from…import 语句，防止污染命名空间

建议20：优先使用 absolute import 来导入模块（Python3中已经移除了relative import）

建议21：i+=1 不等于 ++i，在 Python 中，++i 前边的加号仅表示正，不表示操作

建议22：习惯使用 with 自动关闭资源，特别是在文件读写中

建议23：使用 else 子句简化循环（异常处理）

建议24：遵循异常处理的几点基本原则

（1）注意异常的粒度，try 块中尽量少写代码

（2）谨慎使用单独的 except 语句，或 except Exception 语句，而是定位到具体异常

（3）注意异常捕获的顺序，在合适的层次处理异常

（4）使用更加友好的异常信息，遵守异常参数的规范

建议25：避免 finally 中可能发生的陷阱

建议26：深入理解 None，正确判断对象是否为空。

建议27：连接字符串应优先使用 join 函数，而不是+操作

建议28：格式化字符串时尽量使用 format 函数，而不是 % 形式

建议29：区别对待可变对象和不可变对象，特别是作为函数参数时

建议30：[], {}和()：一致的容器初始化形式。使用列表解析可以使代码更清晰，同时效率更高

建议31：函数传参数，既不是传值也不是传引用，而是传对象或者说对象的引用

建议32：警惕默认参数潜在的问题，特别是当默认参数为可变对象时

建议33：函数中慎用变长参数 args 和 kargs

（1）这种使用太灵活，从而使得函数签名不够清晰，可读性较差

（2）如果因为函数参数过多而是用变长参数简化函数定义，那么一般该函数可以重构

建议34：深入理解 str()和 repr() 的区别

（1）两者之间的目标不同：str 主要面向客户，其目的是可读性，返回形式为用户友好性和可读性都比较高的字符串形式；而 repr 是面向 Python 解释器或者说Python开发人员，其目的是准确性，其返回值表示 Python 解释器内部的定义

（2）在解释器中直接输入变量，默认调用repr函数，而print(var)默认调用str函数

（3）repr函数的返回值一般可以用eval函数来还原对象

（4）两者分别调用对象的内建函数 __str__ ()和 __repr__ ()

建议35：分清静态方法 staticmethod 和类方法 classmethod 的使用场景

## **4. 库的使用**

建议36：掌握字符串的基本用法

建议37：按需选择 sort() 和 sorted() 函数

sort() 是列表在就地进行排序，所以不能排序元组等不可变类型。

sorted() 可以排序任意的可迭代类型，同时不改变原变量本身。

建议38：使用copy模块深拷贝对象，区分浅拷贝（shallow copy）和深拷贝（deep copy）

建议39：使用 Counter 进行计数统计，Counter 是字典类的子类，在 collections 模块中

建议40：深入掌握 ConfigParse

建议41：使用 argparse 模块处理命令行参数

建议42：使用 pandas 处理大型 CSV 文件

Python 本身提供一个CSV文件处理模块，并提供reader、writer等函数。

Pandas 可提供分块、合并处理等，适用于数据量大的情况，且对二维数据操作更方便。

建议43：使用 ElementTree解析XML

建议44：理解模块 pickle 的优劣

优势：接口简单、各平台通用、支持的数据类型广泛、扩展性强

劣势：不保证数据操作的原子性、存在安全问题、不同语言之间不兼容

建议45：序列化的另一个选择 JSON 模块：load 和 dump 操作

建议46：使用 traceback 获取栈信息

建议47：使用 logging 记录日志信息

建议48：使用 threading 模块编写多线程程序

建议49：使用 Queue 模块使多线程编程更安全

## **5. 设计模式**

建议50：利用模块实现单例模式

建议51：用 mixin 模式让程序更加灵活

建议52：用发布-订阅模式实现松耦合

建议53：用状态模式美化代码

## **6. 内部机制**

建议54：理解 build-in 对象

建议55： __init__ ()不是构造方法，理解 __new__ ()与它之间的区别

建议56：理解变量的查找机制，即作用域

局部作用域

全局作用域

嵌套作用域

内置作用域

建议57：为什么需要self参数

建议58：理解 MRO（方法解析顺序）与多继承

建议59：理解描述符机制

建议60：区别 __getattr__ ()与 __getattribute__ ()方法之间的区别

建议61：使用更安全的 property

建议62：掌握元类 metaclass

建议63：熟悉 Python 对象协议

建议64：利用操作符重载实现中缀语法

建议65：熟悉 Python 的迭代器协议

建议66：熟悉 Python 的生成器

建议67：基于生成器的协程和 greenlet，理解协程、多线程、多进程之间的区别

建议68：理解 GIL 的局限性

建议69：对象的管理和垃圾回收

## **7. 使用工具辅助项目开发**

建议70：从 PyPI 安装第三方包

建议71：使用 pip 和 yolk 安装、管理包

建议72：做 paster 创建包

建议73：理解单元测试的概念

建议74：为包编写单元测试

建议75：利用测试驱动开发（TDD）提高代码的可测性

建议76：使用 Pylint 检查代码风格

代码风格审查

代码错误检查

发现重复以及不合理的代码，方便重构

高度的可配置化和可定制化

支持各种 IDE 和编辑器的集成

能够基于 Python 代码生成 UML 图

能够与 Jenkins 等持续集成工具相结合，支持自动代码审查

建议77：进行高效的代码审查

建议78：将包发布到 PyPI

## **8. 性能剖析与优化**

建议79：了解代码优化的基本原则

建议80：借助性能优化工具

建议81：利用 cProfile 定位性能瓶颈

建议82：使用 memory_profiler 和 objgraph 剖析内存使用

建议83：努力降低算法复杂度

建议84：掌握循环优化的基本技巧

减少循环内部的计算

将显式循环改为隐式循环，当然这会牺牲代码的可读性

在循环中尽量引用局部变量

关注内层嵌套循环

建议85：使用生成器提高效率

建议86：使用不同的数据结构优化性能

建议87：充分利用 set 的优势

建议88：使用 multiprocessing 模块克服 GIL 缺陷

建议89：使用线程池提高效率

建议90：使用 Cythonb 编写扩展模块

