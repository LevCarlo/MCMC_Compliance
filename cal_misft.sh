###
 # @Author: Mengjie Zheng
 # @Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
 # @Date: 2024-04-25 10:38:46
 # @LastEditTime: 2024-06-20 12:14:15
 # @LastEditors: Mengjie Zheng
 # @Description: 
 # @FilePath: /Projects/Alaska.Proj/MCMC_Compliance/cal_misft.sh
### 


workdir=/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance
datadir=/Volumes/SeisBig23/mengjie_data/Alaska.Data/COMPLY_INV
logdir=$datadir/LOG
runlst=$workdir/cal_misft.lst

date=$(date +"%Y%m%d")

cat $runlst | parallel -j 20 "echo 'Running: {}'; (time eval '{}') &> $logdir/\`echo {} | awk -F' ' '{print \$NF}'\`.$date.cal_misft.log"