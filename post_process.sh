###
 # @Author: Mengjie Zheng
 # @Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
 # @Date: 2024-04-25 10:38:46
 # @LastEditTime: 2024-06-21 15:56:23
 # @LastEditors: Mengjie Zheng
 # @Description: 
 # @FilePath: /Projects/Alaska.Proj/MCMC_Compliance/post_process.sh
### 


workdir=/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance
datadir=/Volumes/SeisBig23/mengjie_data/Alaska.Data/COMPLY_INV
logdir=$datadir/LOG
runlst=$workdir/post_process.lst

date=$(date +"%Y%m%d")

cat $runlst | parallel -j 10 "echo 'Running: {}'; (time eval '{}') &> $logdir/\`echo {} | awk -F' ' '{print \$NF}'\`.$date.post.process.log"