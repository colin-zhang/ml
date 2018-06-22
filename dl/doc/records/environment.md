### anaconda
[conda document](https://conda.io/docs/user-guide/getting-started.html)

##### 常用命令
`conda info`    
`conda info -e `     
`anaconda-navigator`     
`conda install jupter`    
`conda update jupter`     
删除环境     
`conda remove --name tf --all`    
`conda search numpy`   

##### 修改下载源
修改
```sh
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```
查看
```
$ cat ~/.condarc
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
ssl_verify: true
show_channel_urls: true

```
[清华](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

###### 创建环境
`conda create -n tf tensorflow jupyter python=3.6`      
切换到tf环境    
`source activate tf`    


```
conda create -n pytorch pytorch matplotlib jupyter python=3.6
source activate pytorch
```