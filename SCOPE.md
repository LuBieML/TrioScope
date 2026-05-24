---
tags:
  - System Command
---
<!-- TODO: Unreviewed auto-converted -->
# SCOPE
## Type
_**System Command**_

## Syntax
`SCOPE(enable[, period])`

`SCOPE(enable, period, table_start, table_stop, p0 [,p1 [,p2 [,p3 [,p4 [,p5 [,p6 [,p7]]]]]]])`

`SCOPE(enable, period, fifo_name, fifo_size, samples, p0 [,p1 [,p2 [,p3 [,p4 [,p5 [,p6 [,p7]]]]]]])`

## Description
The SCOPE command enables capture of up to 8 parameters every sample period. Samples are taken until the table range is filled. Trigger is used to start the capture. 

!!! note ""
    _Once loaded the SCOPE must be started by the [TRIGGER](TRIGGER.md) command. [TRIGGER](TRIGGER.md) may specify one-shot or automatic._

!!! danger ""
    Make sure to assign the table range outside of any table data used by your programs.

!!! success ""
    It is normal to use Motion Perfect to assign the SCOPE command, but it is sometimes useful to do it manually. The table data can be read back to a PC and displayed on the Motion Perfect Oscilloscope, saved using Motion Perfect or [STICK_WRITE](STICK_WRITE.md).

## Parameters


|              |                                                  |               |
|--------------|--------------------------------------------------|---------------|
| enable: | 1 or [ON](ON.md) | Enable software SCOPE (requires at least 5 parameters) |
|      __      | 0  or [OFF](OFF.md)                              | Disable SCOPE |
| period:      | The number of servo periods between data samples |       ~~      |
| table_start: | Position to start to store the data in the table array | ~~ |
| table_stop:  | End of table range to use                        |       ~~      |
| fifo_name: | Name of the FIFO file to be written. If this file already exists on the Motion Coordinator it must be a FIFO file. | ~~ |
| fifo_size: | Size of the FIFO file to be created. If the FIFO file already exists with a different size then it will be deleted and recreated with the correct size. | ~~ |
| samples:     | Number of samples to store to the FIFO file.     |       ~~      |
| p0:          | First parameter to store                         |       ~~      |
| p1:          | Second parameter to store                        |       ~~      |
| p2:          | Third parameter to store                         |       ~~      |
| p3:          | Fourth parameter to store                        |       ~~      |
| p4<br>p5 | Fifth parameter to store<br>Sixth parameter to store | ~~ |
| p6           | Seventh parameter to store                       |       ~~      |
| p7           | Eighth parameter to store                        |       ~~      |

## Examples
### Example 1
This example arms the SCOPE to store the [MPOS](MPOS.md) and [DPOS](DPOS.md) on axis 5 axis 5 every 10 milliseconds ([SERVO_PERIOD ](SERVO_PERIOD.md) = 1000). The [MPOS](MPOS.md) will be stored in table values 0 to 499, the [DPOS](DPOS.md) in table values 500 to 999. The sampling does not start until the [TRIGGER](TRIGGER.md) command is executed.

```BASIC
SCOPE(ON, 10, 0, 1000, MPOS AXIS(5), DPOS AXIS(5))
```

### Example 2
Disable the SCOPE to prevent [TRIGGER](TRIGGER.md) from starting a capture

```BASIC
SCOPE(OFF) 
```

### Example 3
Change the period on the fly

```BASIC
SCOPE(ON, 10, 0, 1000, MPOS AXIS(5), DPOS AXIS(5))
TRIGGER
WA(100)
SCOPE(ON, 20) 
```

### Example 4
Delay start of capture until 20 milliseconds after TRIGGER

```BASIC
SCOPE(ON, 10, 0, 1000, MPOS AXIS(5), DPOS AXIS(5))
SCOPE_DELAY = 20
WAIT UNTIL IN(10) = 1
TRIGGER
```

### Example 5
Store 20 ms of data before the trigger.

```BASIC
SCOPE(ON, 10, 0, 1000, MPOS AXIS(5), DPOS AXIS(5))
SCOPE_DELAY = -20
WAIT UNTIL IN(10) = 1
TRIGGER
```

### Example 6
Store 50 samples to the FIFO file “SCOPE_FIFO” and print to terminal \#5.

```BASIC
SCOPE(ON, 1, "SCOPE_FIFO", 2048, 50, TICKS, MPOS AXIS(0), DPOS AXIS(0))
TICKS = 0
TRIGGER
 
OPEN #40 AS "SCOPE_FIFO" FOR FIFO_READ
length = 0
WHILE length < 150
  IF KEY #40 THEN
    GET #40, char
    IF char = 13 THEN
      PRINT #5, ""
      length = length + 1
    ELSE
      PRINT #5, CHR(char);
    ENDIF
    IF char = $2C THEN
      length = length + 1
    ENDIF
  ENDIF
WEND

CLOSE #40
```

The data stored in the FIFO is ASCII CSV format.  Each row has the data for one capture point. For example:

```BASIC
0.00000,45.00000,45.00000
-1.00000,47.00000,47.00000
-2.00000,51.00000,51.00000
-3.00000,56.00000,55.00000
-4.00000,60.00000,60.00000
-5.00000,63.00000,62.00000
```

## See Also
[TRIGGER](TRIGGER.md), [SCOPE_POS](SCOPE_POS.md), [SCOPE_CYCLE_COUNT](SCOPE_CYCLE_COUNT.md), [SCOPE_DELAY](SCOPE_DELAY.md)

