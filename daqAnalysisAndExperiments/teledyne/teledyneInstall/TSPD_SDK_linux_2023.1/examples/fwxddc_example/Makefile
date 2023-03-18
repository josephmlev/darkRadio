all: CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX -Wall -Wextra
all: LIBS += $(shell pkg-config --libs libadq)
all: adqapifwxddcexample
debug: CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX -Wall -Wextra
debug: LIBS += $(shell pkg-config --libs libadq)

local: CFLAGS += -DLINUX
local: LIBS += -ladq -ludev -lpthread -lstdc++
local: ADQAPI_FWxDDC_example.c
	$(CC) $(CFLAGS) -I../../Release -o adqapifwxddcexample ADQAPI_FWxDDC_example.c -L../../source/.libs $(LIBS)

debug: ADQAPI_FWxDDC_example.c
	$(CC) $(CFLAGS) -g -o adqapifwxddcexample ADQAPI_FWxDDC_example.c $(LIBS)

adqapifwxddcexample: ADQAPI_FWxDDC_example.c
	$(CC) $(CFLAGS) -o adqapifwxddcexample ADQAPI_FWxDDC_example.c $(LIBS)

clean:
	rm -f adqapifwxddcexample
