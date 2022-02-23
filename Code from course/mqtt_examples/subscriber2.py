from DoSomething import DoSomething
import time


if __name__ == "__main__":
	test = DoSomething("subscriber 2")
	test.run()
	test.myMqttClient.mySubscribe("/my/first/topic")

	a = 0
	while (a < 30):
		a += 1
		time.sleep(1)

	test.end()