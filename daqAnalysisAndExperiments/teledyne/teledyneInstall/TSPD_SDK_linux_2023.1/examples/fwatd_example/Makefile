CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX -Wall -Wextra -Wpedantic
LIBS += $(shell pkg-config --libs libadq)
SOURCES = example_adq7.c \
    example_adq14.c \
    ADQAPI_FWATD_example.c \
    utils.c

all: adqapifwatdexample
debug: adqapifwatdexample
local: adqapifwatdexample

local: CFLAGS += -g -I../../Release -Wall -Wextra -Wpedantic
local: LOCAL_LIBS += -L../../source/.libs
debug: CFLAGS += -g -Wall -Wextra -Wpedantic

OBJECTS = $(SOURCES:.c=.o)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

adqapifwatdexample: $(OBJECTS)
	$(CC) $(CFLAGS) -o adqapifwatdexample $(OBJECTS) $(LOCAL_LIBS) $(LIBS)

clean:
	rm -f adqapifwatdexample $(OBJECTS)
