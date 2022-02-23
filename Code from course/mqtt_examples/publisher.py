from DoSomething import DoSomething
import time


if __name__ == "__main__":
	test = DoSomething("publisher 1")
	test.run()

	a = 0
	while (a < 20):
		test.myMqttClient.myPublish ("/my/first/topic", ("my message %d" % (a))) 	
		a += 1
		time.sleep(1)

	test.end()