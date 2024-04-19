###
 # @Author: Mengjie Zheng
 # @Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
 # @Date: 2023-07-11 18:35:25
 # @LastEditors: Mengjie Zheng
 # @LastEditTime: 2023-11-01 20:49:50
 # @FilePath: /Projects/Alaska.Proj/inv_inversion/MC_Compliance-dev/run.sh
 # @Description: 
### 


workdir=/Users/mengjie/Projects/Alaska.Proj/inv_inversion/MC_Compliance-dev
datadir=/Volumes/Tect32TB/Mengjie/Alaska.Data
logdir=$datadir/LOG/Inversion
runlst=$workdir/run.lst

date=$(date +"%Y%m%d")
# cat $runlst | parallel -j 8 "echo 'Running: {}'; eval '{}; (time {})' &> $logdir/\`echo {} | awk -F' ' '{print \$NF}'\`.$date.INV.log"

cat $runlst | parallel -j 8 "echo 'Running: {}'; (time eval '{}') &> $logdir/\`echo {} | awk -F' ' '{print \$NF}'\`.$date.INV.log"
