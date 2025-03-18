# Robotics Intro

> Note Taking: Alex
>
> Reference: `https://fishros.com/d2lros2/#/humble/` 小鱼**ros**教程

## ROS2

#### 说在前面的话

可以一键安装很多软件：

```bash
wget http://fishros.com/install -O fishros && . fishros
```

一路按照说明进行安装，我们这里的版本是`ros2/humble`

笔者踩了许多坑，在这里总结入门ros2的最简单方法。

安装好了ros2后，开机启动

```bash
source ~/.bashrc
```

`Helloworld`程序：打开两个`Ubuntu`终端，分别输入：

```bash
ros2 run demo_nodes_py listener
```

```bash
ros2 run demo_nodes_cpp talker
```

可以看到如下的两个终端的`talker`和`listener`的内容

![屏幕截图 2025-03-18 201805](assets/屏幕截图 2025-03-18 201805.png)

传统的机器人学中，机器人分为**感知，决策，控制**三部分。Ros系统的基本组成是Node，Node之间通过通信协议完成机器人的功能。

![image-20220602204152352](https://fishros.com/d2lros2/humble/chapt1/advanced/1.ROS2%E7%B3%BB%E7%BB%9F%E6%9E%B6%E6%9E%84/imgs/image-20220602204152352.png)

![Nodes-TopicandService](https://fishros.com/d2lros2/humble/chapt2/get_started/1.ROS2%E8%8A%82%E7%82%B9%E4%BB%8B%E7%BB%8D/imgs/Nodes-TopicandService-16542449255392.gif)

