#import libraries
import RPi.GPIO as GPIO
import time

#GPIO Basic initialization
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#Use a variable for the Pin to use
#If you followed my pictures, it's port 7 => BCM 4
led = 4
cnt = 0
#Initialize your pin
GPIO.setup(led,GPIO.OUT)

while cnt < 4:
#Turn on the LED
	print("LED on")
	GPIO.output(led,1)

	#Wait 5s
	
	time.sleep(5)
	cnt = cnt+1

	#Turn off the LED
	print("LED off")
	GPIO.output(led,0)
	time.sleep(5)
