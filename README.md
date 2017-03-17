# try-keras-mnist

keras で mnist の学習をさせてみる簡単な例

## Docker で環境を作る

このレポジトリをクローンしてから、 keras がすぐに使える Docker コンテナを立てる。

```sh
$ git clone https://github.com/fujiharuka/try-keras-mnist.git
$ cd try-keras-mnist
$ sudo docker run -it -v $(pwd):/srv gw000/keras:1.2.2-py3 bash
```

## 学習させてみる

上記の Docker コンテナ内で、すぐ実行！

```sh
/srv# python3 learn.py
```
