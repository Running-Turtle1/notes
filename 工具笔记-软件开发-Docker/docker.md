## 常用命令

1. **删除容器/镜像**

- 删除镜像：rmi
- 删除容器：rm

2. **docker run 参数**

- -d：后台运行
- -p：p1/p2 宿主机端口/容器端口
- -v：volume1/volume2 绑定挂载目录
  - 注意宿主机目录会覆盖容器的目录

- 命名卷挂载
  - 第一次使用时，容器的内容会被同步到宿主机
- -e 传递环境变量
- --name 起名字（unique）
- -it 进入容器内部交互
- --rm 退出容器时删除，用于临时调试容器（常与 -it 搭配）
- --restart 配置重启策略
  - always：容器停止了就立即重启（或者换成 unless-stopped，这个不会重启手动停止的容器）

3. **docker start**

- 第二次启动就不需要重新传参数了
  - docker inspect \<imageid\> 可以重新查看参数（贴给AI就行）

4. docker logs --follow 容器名

滚动查看日志

## 推送镜像至 dockerhub

## 创建子网

桥接

host

none

子网

## docker compose 容器编排

`docker-compose.yaml`

docker 给每个 compose 创建一个子网

可以控制容器启动顺序

（AI-ERA, 把需要执行的命令给AI即可生成 compose 文件）

