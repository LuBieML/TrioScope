using OpenTK.Audio.OpenAL;
using ScottPlot.Palettes;
using System;
using System.ComponentModel;
using System.IO;
using System.Reflection;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Contexts;
using System.Threading;
using Trio.UnifiedApi;

namespace docAutomation
{
    static class Constants
    {
        public const string VERSION_SW = "scope 0.1";
        public const bool debug = true;        // 1 activate debug messages
        public const int data_table_len = 8192;     // uso 4 ch e non so perche'

    }
    class UapiHistance : IDisposable
    {
        ITrioConnection _connection;
        public int flowState = 0;
        int tableSize = 1000;
        public double[] dataT = new double[Constants.data_table_len];
        public double[] dataX = new double[Constants.data_table_len];
        public double[] dataY = new double[Constants.data_table_len];
        public double[] dataY1 = new double[Constants.data_table_len];
        public double[] dataY2 = new double[Constants.data_table_len];
        public double[] dataY3 = new double[Constants.data_table_len];
        public double[] dataY4 = new double[Constants.data_table_len];
        public double[] dataY5 = new double[Constants.data_table_len];
        public double[] dataY6 = new double[Constants.data_table_len];
        public double[] dataY7 = new double[Constants.data_table_len];


        public String trigger_source = "speed fbk";
        public String trigger_mode = "none";
        public int trigger_preTrigger = 100;
        public double triggerLevel = 10;

        public int debugVar = 0;


        public String outFileName;
        public String onlyFileName;
        string docPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        String recordFolder = "C:\\Users\\giorgio.rancilio\\Documents\\aaa\\003_applicazioni\\17_utility\\100_CSharp\\docAutomation\\docAutomation\\docOutput\\records\\";

        public UapiHistance(string mcIP)
        {
            _connection = string.Equals(mcIP, "PCMCAT", StringComparison.OrdinalIgnoreCase) ?
                                TrioConnectionFactory.CreateConnectionPCMCAT(ConnectionCallback) :
                          string.Equals(mcIP, "Flex7", StringComparison.OrdinalIgnoreCase) ?
                                TrioConnectionFactory.CreateConnectionFlex7(ConnectionCallback) :
                                TrioConnectionFactory.CreateConnectionTCP(ConnectionCallback, mcIP);
            Console.WriteLine("UapiHistance: created. " + mcIP);

        }
        ~UapiHistance()
        {
            Dispose();
        }

        public void Dispose()
        {
            var connection = Interlocked.Exchange(ref _connection, null);
            connection?.Dispose();
        }

        public void Execute(string cmd, int _sampleTime, int axisIndex, string _driveModel, int deviceAddrees)
        {
            var conn = _connection;
            String cmdStr;

            Console.WriteLine("cmd :" + cmd);
            //int axisIndex = 0;

            switch (cmd)
            {
                case "writeSpeed":
                    if (conn == null)
                    {
                        throw new TrioConnectionException(ErrorCode.ConnectionContext);
                    }
                    conn.OpenConnection();
                    // do stuffs
                    //Set AxisParamter - SPEED
                    double speed = 2;
                    conn.SetAxisParameter_SPEED(axisIndex, speed);
                    //conn.Forward()
                    //Get AxisParameter - SPEED
                    double speedRd = conn.GetAxisParameter_SPEED(axisIndex);
                    if (speedRd != speed)
                    {
                        Console.WriteLine($"Error getting AxisParameter value - SPEED");
                    }


                    // then close
                    conn.CloseConnection();
                    break;

                case "waitEOA":
                    if (conn == null)
                    {
                        throw new TrioConnectionException(ErrorCode.ConnectionContext);
                    }
                    conn.OpenConnection();

                    double valRd = 0;


                    while (valRd != 3)
                    {
                        Thread.Sleep(200);
                        // read scope status
                        cmdStr = "co_read_axis(" + Convert.ToString(axisIndex) + ",$2066 , 0, 4, 99)";
                        conn.Execute(cmdStr);
                        Thread.Sleep(200);
                        //Get VR value
                        valRd = conn.GetVrValue(99);

                    }
                    // then close
                    conn.CloseConnection();
                    flowState = 9;
                    break;


                case "setScope":

                    flowState = 1;
                    if (conn == null)
                    {
                        throw new TrioConnectionException(ErrorCode.ConnectionContext);
                    }
                    conn.OpenConnection();
                    // do stuffs

                    if (_driveModel == "DX3")
                    {
                        /*
                            0x368c.1    = 0         //trigger mode. 0 No trg
                            0x368c.7    = 1         // sample time, unit 125us
                            0x368c.8    = 0x0F10    // spd fbk cosa ch1
                            0x368c.9    = 0x0F20    // IQ_REF_ADDR  ch2
                            0x368c.            // cosa ch1
                            0x368c.8            // cosa ch1

                            0x368B	    0: Stop sample 1: Start sample


                            0x3680
                                    Bit0~bit13	reserved	
                                    Bit14:bit15	0:not in sampling status
                                    1:sampling
                                    2:sampling is done	RO

                        */

                        //conn.Execute("co_write_axis(0, $368b, 0, 6, -1,    0)");    // stop triggering
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368b, 0, 6, -1,    0)";
                        conn.Execute(cmdStr);
                        //conn.Execute("co_write_axis(0, $368c, 1, 6, -1,    0)");    // no trigger
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 1, 6, -1,    0)";
                        conn.Execute(cmdStr);
                        //conn.Execute("co_write_axis(0, $368c, 7, 6, -1,    1)");    // sample time
                        String setSampleTime = "co_write_axis(0, $368c, 7, 6, -1," + Convert.ToString(_sampleTime) + ")";
                        conn.Execute(setSampleTime);    // sample time

                        //conn.Execute("co_write_axis(0, $368c, 8, 6, -1, $f10)");
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 8, 6, -1, $f10)";
                        conn.Execute(cmdStr);

                        //conn.Execute("co_write_axis(0, $368c, 9, 6, -1, $f20)");
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 9, 6, -1, $f20)";
                        conn.Execute(cmdStr);

                        // added position
                        //conn.Execute("co_write_axis(0, $368c, 10, 6, -1, 0)");
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 10, 6, -1, $0f16)";
                        conn.Execute(cmdStr);
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 11, 6, -1, $0f17)";
                        conn.Execute(cmdStr);
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 12, 6, -1, $0f18)";
                        conn.Execute(cmdStr);
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 13, 6, -1, $0f19)";
                        conn.Execute(cmdStr);
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 14, 6, -1, 0)";
                        conn.Execute(cmdStr);
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368c, 15, 6, -1, 0)";

                        //conn.Execute("co_write_axis(0, $368b, 0, 6, -1,    1)");    // start acquiring
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368b, 0, 6, -1, 1)";
                        conn.Execute(cmdStr);

                        //TODO: wait end of acq DX3

                        //                    while( (conn.Ethercat_CoReadAxis(0, 0x3680, 0, Co_ObjectType.Unsigned16, -1) && 0xC000) == 0xc00)

                        //conn.Ethercat_CoReadAxis
                        /*
                        conn.Ethercat_CoWriteAxis_Value(0, 0x368c, 1, Co_ObjectType.Unsigned16, 0);
                        conn.Ethercat_CoWriteAxis_Value(0, 0x368c, 7, Co_ObjectType.Unsigned16, 1);

                        conn.Ethercat_CoWriteAxis_Value(0, 0x368c, 8, Co_ObjectType.Unsigned16, 0x0f10);
                        conn.Ethercat_CoWriteAxis_Value(0, 0x368c, 9, Co_ObjectType.Unsigned16, 0x0f20);

                        conn.Ethercat_CoWriteAxis_Value(0, 0x368B, 0, Co_ObjectType.Unsigned16, 1); // start sample
                        */
                    }
                    else
                    {
                        // DX5/DX1

                        // according axis odd or even the channel object offset change
                        // 0x800
                        // Dx5 is a double axis drive. it uses one address and two axis numeber
                        // DX1 is single axis. the second axis doesn't exit. 

                        int chOffset;
                        if (_driveModel == "DX1") chOffset = 0;
                        else
                        {
                            if (axisIndex % 2 == 0) chOffset = 0;
                            else chOffset = 0x800;
                        }
                        int sleep_time = 50;
                        int objAdd;

                        //$2065:00    $00000004   $000000FF Z_AScopeEnable = 0
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ",$2065 , 0, 4, -1,    0)";
                        conn.Execute(cmdStr);
                        Thread.Sleep(sleep_time);

                        // $2066:00    $00000004   $00000087   Z_AScopeStatus
                        //cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $368b, 0, 6, -1,    0)";
                        //conn.Execute(cmdStr);
                        // $2067:00    $00000004   $00000087   Z_AScopeCount counter read only
                        // $2068:00    $00000004   $000000FF Z_AScopeSamples (length)
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2068, 0, 4, -1,  8000)";
                        conn.Execute(cmdStr);
                        Thread.Sleep(sleep_time);

                        // $2069:00    $00000004   $000000FF Z_AScopeSteps
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2069, 0, 4, -1," + Convert.ToString(_sampleTime) + ")";
                        conn.Execute(cmdStr);
                        Thread.Sleep(sleep_time);

                        // $206A: 00    $00000004   $000000FF Z_AScopeTrigChan 0 no trigger
                        // Number of channel used to evaluate trigger.Set to 0 to disable trigger
                        /*
                            trigger_source
                            none
                            speed fbk
                            Iqr
                            
                            trigger_mode
                            greater than 0 up
                            lower than   1 down

                        */
                        if      (trigger_source == "none")       cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206A, 0, 4, -1,    0)";
                        else if (trigger_source == "speed fbk")  cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206A, 0, 4, -1,    1)";   // ch1
                        else if (trigger_source == "onAlarm")    cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206A, 0, 4, -1,    5)"; // active alarm
                        else                                     cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206A, 0, 4, -1,    2)";   // ch2 (trigger_source == "Iqr")

                        //                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206A, 0, 4, -1,    1)";
                        conn.Execute(cmdStr);
                        Thread.Sleep(sleep_time);

                        // $206B: 00    $00000008   $000000FF Z_AScopeTrigVal
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206B, 0, 8, -1," + Convert.ToString(triggerLevel) +")";
                        conn.Execute(cmdStr);
                        Thread.Sleep(sleep_time);

                        // $206C: 00    $00000004   $000000FF Z_AScopeTrigUpDown
                        if      (trigger_mode == "greater than")    cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206C, 0, 4, -1,    0)";
                        else                                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206C, 0, 4, -1,    1)";

                        //cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206C, 0, 4, -1,    0)";
                        conn.Execute(cmdStr);
                        Thread.Sleep(sleep_time);

                        // $206D:00    $00000004   $000000FF Z_AScopeTrigPre (pretrigger)
                        cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206D, 0, 4, -1," +  trigger_preTrigger + ")";
                        conn.Execute(cmdStr);
                        Thread.Sleep(sleep_time);

                        if (debugVar == 0)
                        {

                            // $206E:00    $00000004   $000000FF Z_AScopeCh1
                            objAdd = 0x36df + chOffset;// speed_fbk 
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206E, 0, 4, -1, " + Convert.ToString(objAdd) + ")";
                            conn.Execute(cmdStr);
                            Thread.Sleep(sleep_time);

                            // $206F:00    $00000004   $000000FF Z_AScopeCh2
                            objAdd = 0x36eb + chOffset;// iqr
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $206F, 0, 4, -1, " + Convert.ToString(objAdd) + ")";
                            conn.Execute(cmdStr);
                            Thread.Sleep(sleep_time);

                            // $2070:00    $00000004   $000000FF Z_AScopeCh3
                            objAdd = 14064 + chOffset;//A_Position Feedback
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2070, 0, 4, -1, " + Convert.ToString(objAdd) + ")";
                            conn.Execute(cmdStr);
                            Thread.Sleep(sleep_time);

                            // $2071:00    $00000004   $000000FF Z_AScopeCh4
                            objAdd = 14050 + chOffset;//A_Position Feedback
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2071, 0, 4, -1, " + Convert.ToString(objAdd) + ")";
                            conn.Execute(cmdStr);
                            Thread.Sleep(sleep_time);

                            // $2072:00    $00000004   $000000FF Z_AScopeCh5 
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2072, 0, 4, -1,    14392)"; // active alarm
                            conn.Execute(cmdStr);
                            Thread.Sleep(sleep_time);

                            // $2073:00    $00000004   $000000FF Z_AScopeCh6
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2073, 0, 4, -1,    0)";
                            conn.Execute(cmdStr);
                            Thread.Sleep(sleep_time);

                            // $2074:00    $00000004   $000000FF Z_AScopeCh7
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2074, 0, 4, -1,    0)";
                            conn.Execute(cmdStr);
                            Thread.Sleep(sleep_time);

                            // $2075:00    $00000004   $000000FF Z_AScopeCh8
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2075, 0, 4, -1,    0)";
                            conn.Execute(cmdStr);
                            Thread.Sleep(sleep_time);

                        }
                        else
                        {   
                            // set on the 8 channels the DBG float 0 to 7
                            int IPA = 0x206E;
                            int IPAtoSample = 13921;
                            for( int  i= 0; i < 8; i++)
                            {
                                cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", " + Convert.ToString(IPA+i) +", 0, 4, -1, " + Convert.ToString(IPAtoSample+i)    +")";
                                conn.Execute(cmdStr);
                                Thread.Sleep(sleep_time);

                            }
                        }
                            //$2065:00    $00000004   $000000FF Z_AScopeEnable
                            cmdStr = "co_write_axis(" + Convert.ToString(axisIndex) + ", $2065, 0, 4, -1,    1)";
                        conn.Execute(cmdStr);
                        Thread.Sleep(sleep_time);
                    }

                    // Ethercat_CoRead_IntValue
                    //TODO: wait end of acq DX5
                    // Ethercat_CoRead_IntValue
                    // come faccio la read??
                    /*
                     void Trio.UnifiedApi.ITrioConnection.Ethercat_CoReadAxis	(	Int32	axis_number,
 	 	UInt16	object_index,
 	 	Byte	object_subindex,
 	 	Co_ObjectType	type,
 	 	Int32	out_vr_index ) 
                    */

                        // then close
                        flowState = 2;

                    conn.CloseConnection();
                    break;

                case "drv2mc":
                    flowState = 3;
                    if (conn == null)
                    {
                        throw new TrioConnectionException(ErrorCode.ConnectionContext);
                    }
                    conn.OpenConnection();

                    if (_driveModel == "DX3")
                    {


                        /*
                         * vedi doc in C:\Users\giorgio.rancilio\Documents\aaa\003_applicazioni\04_fieraVibrazioni\01_notch_doc
                         * IPD-PLN-T22 COMBO-function design documentV1.0_20200120 1.docx
                         * 
                         */
                        // Initiate the EC_COE_FIFO file transfer from the PC-MCAT.
                        long device_address = 1;
                        long max_size = 16000; // scope dimension in bytes
                        int response_size = 256;
                        string response;

                        //conn.Execute("ethercat($161, 0, 1, $3687, 0, 16000)");
                        cmdStr = "ethercat($161, 0, " + Convert.ToString(axisIndex + 1) + ", $3687, 0, 16000)";
                        conn.Execute(cmdStr);

                        Thread.Sleep(2000);
                    }
                    else if (_driveModel == "DX5")
                    {

                        cmdStr = "ETHERCAT($141, 0," + Convert.ToString(deviceAddrees) + @",""C"", ""EC_COE_FIFO"", ""ASCOPE_data0"", -1) ";
                        conn.Execute(cmdStr);

                        //TODO: wait end of transfer file
                        /*                       while (int progress < 100)
                                               {
                                                   progress = ETHERCAT( $142);
                                               }
                        */
                        Thread.Sleep(5000);
                    }
                    else if (_driveModel == "DX1")
                    {

                        cmdStr = "ETHERCAT($141, 0," + Convert.ToString(axisIndex + 1) + @",""C"", ""EC_COE_FIFO"", ""ASCOPE_data0"", -1) ";
                        conn.Execute(cmdStr);
                        //TODO: wait end of transfer file
                        /*                       while (int progress < 100)
                                               {
                                                   progress = ETHERCAT( $142);
                                               }
                        */
                        Thread.Sleep(5000);
                    }
                    else Console.WriteLine("worng drive model: " + _driveModel);
                    // then close
                    conn.CloseConnection();

                    flowState = 4;

                    break;


                case "getFile":
                    flowState = 5;

                    if (conn == null)
                    {
                        throw new TrioConnectionException(ErrorCode.ConnectionContext);
                    }
                    conn.OpenConnection();
                    /*
                     * vedi doc in C:\Users\giorgio.rancilio\Documents\aaa\003_applicazioni\04_fieraVibrazioni\01_notch_doc
                     * IPD-PLN-T22 COMBO-function design documentV1.0_20200120 1.docx
                     * 
                     */

                    conn.DownloadFile("locale.bin", "EC_COE_FIFO", _handler);

                    Thread.Sleep(1000);

                    /*
                    if (conn.ethe(new Int64[] { 0x161, 0, device_address, 0x3687, 0, max_size }, out response, response_size) != 0)
                    {
                        Console.WriteLine("Ethercat transfer failed");
                        throw new Exception("EC_COE_FIFO file transfer failed");
                    }
                    */
                    // spacchetta dati
                    string path = @"locale.bin";

                    if (_driveModel == "DX3")
                    {
                        // Calling the ReadAllBytes() function 
                        byte[] readText = File.ReadAllBytes(path);

                        int i, j;
                        //                 for (i = 0, j = 0; j < Constants.data_table_len; i += 4, j++)
                        tableSize = readText.Length;
                        tableSize = (int)Math.Floor((double)(tableSize / 16));
                        // il drive ritorna più dati dei necessari.
                        // occorre vedere solo i primi 1000 
                        tableSize = 1000;
                        int colonne = 6;//dati a 16 bit,  1 spd, 1 Iqr, 4 pos // 2;
                        Console.WriteLine("tableSize: " + tableSize);


                        onlyFileName = Convert.ToString(axisIndex) + "_" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "record";
                        outFileName = recordFolder + onlyFileName + ".txt";

                        //                    using (StreamWriter outputFile = new StreamWriter(Path.Combine(docPath, outFileName), true)) // write in Documents path
                        using (StreamWriter outputFile = new StreamWriter(outFileName, true))
                        {
                            outputFile.WriteLine("axisIndex:\t" + Convert.ToString(axisIndex) + "\tFileName:\t" + outFileName); // header row
                            outputFile.WriteLine("time[ms]\t spdFbk[rpm]\t Iqr[0.1%In]\t Position[64bit]"); // header row
                            for (i = 0, j = 0; j < tableSize; j++, i += colonne * 2) // 2 byte per dato
                            {
                                dataX[j] = j;
                                dataT[j] = j * _sampleTime * 125e-3; // metto t in ms
                                                                     //                            short tmp = (short)(readText[i + 1] << 8 | readText[i + 0]);
                                                                     //                            tmp = (short)(readText[i + 3] << 8 | readText[i + 2]);
                                dataY[j] = (short)(readText[i + 1] << 8 | readText[i + 0]);
                                dataY1[j] = (short)(readText[i + 3] << 8 | readText[i + 2]);
                                dataY2[j] = (long)(readText[i + 11] << 56 | readText[i + 10] << 48 | readText[i + 9] << 40 | readText[i + 8] << 32 | readText[i + 7] << 24 | readText[i + 6] << 16 | readText[i + 5] << 8 | readText[i + 4]);

                                //                            dataY1[j] =((int)readText[i + 2] + (255 * (int)readText[i + 3]));
                                /*                        dataY2[j] = (int)readText[i + 4] + (255 * (int)readText[i + 5]);
                                                        dataY3[j] = (int)readText[i + 6] + (255 * (int)readText[i + 7]);
                                                        dataY4[j] = (int)readText[i + 8] + (255 * (int)readText[i + 9]);
                                                        dataY5[j] = (int)readText[i +10] + (255 * (int)readText[i +11]);
                                                        dataY6[j] = (int)readText[i +12] + (255 * (int)readText[i +13]);
                                                        dataY7[j] = (int)readText[i +14] + (255 * (int)readText[i +15]);

                                 */
                                //Console.WriteLine("{0}\t {1}\t {2}\t {3}", (int)readText[i + 0], (int)readText[i + 1], (int)readText[i + 2], (int)readText[i + 3]);
                                //Console.WriteLine("{0}\t {1}\t {2}\t {3}\t {4}\t {5}\t {6}\t {7}", j, dataY[j], dataY1[j], dataY2[j], dataY3[j], dataY4[j], dataY5[j], dataY6[j], dataY7[j]);
                                //Console.WriteLine("{0}\t {1}\t {2}\t", j, dataY[j], dataY1[j]);
                                outputFile.WriteLine("{0}\t {1}\t {2}\t {3}\t", dataT[j], dataY[j], dataY1[j], dataY2[j]);


                            }
                        }
                    }
                    else
                    {
                        // DX5 DX1
                        //AScope2Data.exe ASCOPE_data0.efw data.csv
                        // convert data to CSV
                        string currentPath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);

                        using (System.Diagnostics.Process pProcess = new System.Diagnostics.Process())
                        {
                            pProcess.StartInfo.FileName = currentPath + @"\AScope2DataDx5.exe";
                            pProcess.StartInfo.Arguments = "locale.bin data.csv"; //argument
                            pProcess.StartInfo.UseShellExecute = false;
                            pProcess.StartInfo.RedirectStandardOutput = true;
                            pProcess.StartInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
                            pProcess.StartInfo.CreateNoWindow = true; //not diplay a windows
                            pProcess.Start();
                            string output = pProcess.StandardOutput.ReadToEnd(); //The output result
                            pProcess.WaitForExit();
                        }



                        string InFileName = @"data.csv";
                        int lineNumber = 0;

                        onlyFileName = Convert.ToString(axisIndex) + "_" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "record";
                        outFileName = recordFolder + onlyFileName + ".txt";

//                        outFileName = recordFolder + Convert.ToString(axisIndex) + "_" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "record.txt";
                        //                        outFileName = Convert.ToString(axisIndex) + "_" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "record.txt";
                        using (StreamWriter outputFile = new StreamWriter(outFileName, true))
                        {
                            outputFile.WriteLine("axisIndex:\t" + Convert.ToString(axisIndex) + "\tFileName:\t" + outFileName); // header row
                            outputFile.WriteLine("time[ms]\t spdFbk[rpm]\t Iqr[0.1%In]\t Position[64bit]"); // header row

                            using (StreamReader reader = new StreamReader(InFileName))
                            {
                                string line;
                                int j = 0;
                                while ((line = reader.ReadLine()) != null)
                                {
                                    lineNumber++;
                                    Console.WriteLine(line);
                                    if (lineNumber > 8)
                                    {
                                        var fields = line.Split('\t');


                                        dataX[j] = j;
                                        if (_driveModel == "DX5")   dataT[j] = j * _sampleTime *  125e-3; // metto t in ms
                                        else                        dataT[j] = j * _sampleTime * 62.5e-3; // metto t in ms
                                        dataY[j] = Convert.ToDouble(fields[1]);
                                        dataY1[j] = Convert.ToDouble(fields[2]);
                                        dataY2[j] = Convert.ToDouble(fields[3]);
                                        dataY3[j] = Convert.ToDouble(fields[4]);

                                        outputFile.WriteLine("{0}\t {1}\t {2}\t {3}\t {4}\t", dataT[j], dataY[j], dataY1[j], dataY2[j], dataY3[j]);
                                        j++;

                                    }
                                }
                            }

                        }
                    }

                    // then close
                    flowState = 6;

                    conn.CloseConnection();

                    break;



            }// end case



            /*****************Writing and reading variables*****************/


            /***********************Disconnect from MC***********************/

            //            conn.CloseConnection();
        }


        // mostra avanzamento
        // potrebbe intercettare errori o altro
        private void _handler(ref ProgressInfo progress)
        {
            Console.WriteLine("bytes: " + progress.current_pos.ToString());
            return;
            throw new NotImplementedException();
        }







        void ConnectionCallback(EventType event_type, long int_value, string str_value)
        {
            switch (event_type)
            {
                case EventType.Error:
                    Console.WriteLine($"Error [0x{int_value:X}] occurred: {str_value}");
                    break;

                case EventType.Warning:
                    Console.WriteLine($"Warning [0x{int_value:X}] occurred: {str_value}");
                    break;

                case EventType.Message:
                    Console.WriteLine($"Msg: {str_value}");
                    break;
            }
        }
    }

}
