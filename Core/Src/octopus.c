/*
 * octopus.c
 *
 *  Created on: Mar 2, 2023
 *      Author: lenovo
 */

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "usart.h"
#include "string.h"
#include "octopus.h"
#include "flash.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */
//extern uint16_t ADC_Value[4];

const int num_tab[10] = { 1000000000, 100000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10, 1 };
const unsigned char crc8_table[] = { //x8+x5+x4+1
		0x00, 0x5E, 0xBC, 0xE2, 0x61, 0x3F, 0xDD, 0x83, 0xC2, 0x9C, 0x7E, 0x20, 0xA3, 0xFD, 0x1F, 0x41, 0x9D, 0xC3, 0x21, 0x7F, 0xFC, 0xA2, 0x40, 0x1E, 0x5F,
				0x01, 0xE3, 0xBD, 0x3E, 0x60, 0x82, 0xDC, 0x23, 0x7D, 0x9F, 0xC1, 0x42, 0x1C, 0xFE, 0xA0, 0xE1, 0xBF, 0x5D, 0x03, 0x80, 0xDE, 0x3C, 0x62, 0xBE,
				0xE0, 0x02, 0x5C, 0xDF, 0x81, 0x63, 0x3D, 0x7C, 0x22, 0xC0, 0x9E, 0x1D, 0x43, 0xA1, 0xFF, 0x46, 0x18, 0xFA, 0xA4, 0x27, 0x79, 0x9B, 0xC5, 0x84,
				0xDA, 0x38, 0x66, 0xE5, 0xBB, 0x59, 0x07, 0xDB, 0x85, 0x67, 0x39, 0xBA, 0xE4, 0x06, 0x58, 0x19, 0x47, 0xA5, 0xFB, 0x78, 0x26, 0xC4, 0x9A, 0x65,
				0x3B, 0xD9, 0x87, 0x04, 0x5A, 0xB8, 0xE6, 0xA7, 0xF9, 0x1B, 0x45, 0xC6, 0x98, 0x7A, 0x24, 0xF8, 0xA6, 0x44, 0x1A, 0x99, 0xC7, 0x25, 0x7B, 0x3A,
				0x64, 0x86, 0xD8, 0x5B, 0x05, 0xE7, 0xB9, 0x8C, 0xD2, 0x30, 0x6E, 0xED, 0xB3, 0x51, 0x0F, 0x4E, 0x10, 0xF2, 0xAC, 0x2F, 0x71, 0x93, 0xCD, 0x11,
				0x4F, 0xAD, 0xF3, 0x70, 0x2E, 0xCC, 0x92, 0xD3, 0x8D, 0x6F, 0x31, 0xB2, 0xEC, 0x0E, 0x50, 0xAF, 0xF1, 0x13, 0x4D, 0xCE, 0x90, 0x72, 0x2C, 0x6D,
				0x33, 0xD1, 0x8F, 0x0C, 0x52, 0xB0, 0xEE, 0x32, 0x6C, 0x8E, 0xD0, 0x53, 0x0D, 0xEF, 0xB1, 0xF0, 0xAE, 0x4C, 0x12, 0x91, 0xCF, 0x2D, 0x73, 0xCA,
				0x94, 0x76, 0x28, 0xAB, 0xF5, 0x17, 0x49, 0x08, 0x56, 0xB4, 0xEA, 0x69, 0x37, 0xD5, 0x8B, 0x57, 0x09, 0xEB, 0xB5, 0x36, 0x68, 0x8A, 0xD4, 0x95,
				0xCB, 0x29, 0x77, 0xF4, 0xAA, 0x48, 0x16, 0xE9, 0xB7, 0x55, 0x0B, 0x88, 0xD6, 0x34, 0x6A, 0x2B, 0x75, 0x97, 0xC9, 0x4A, 0x14, 0xF6, 0xA8, 0x74,
				0x2A, 0xC8, 0x96, 0x15, 0x4B, 0xA9, 0xF7, 0xB6, 0xE8, 0x0A, 0x54, 0xD7, 0x89, 0x6B, 0x35 };

const STRUCT_Shell_Commands mv_shell_cmds[] = {
		{ "test", (shell_cmd_func)   cmd_Test, "for test" },
		{ "help", (shell_cmd_func)   cmd_Displayhelp, "help & usages" },
		{ "getAdc", (shell_cmd_func) cmd_GetAdcValue, "get adc convert value" },
		{ NULL, NULL,NULL },
};

uint8_t crc8(uint8_t *buf, uint16_t len) {
	unsigned char bl = 0;
	unsigned char crc8 = 0;
	for (bl = 0; bl < len; bl++) {
		crc8 = crc8_table[crc8 ^ *(buf + bl)];
	}
	return (crc8);
}

uint16_t MakeWord(uint8_t b1, uint8_t b0) {
	uint16_t value = b1;
	value = (value << 8) + b0;
	return value;
}

//比如字符串"1234567890" 返回1234567890数值
uint32_t char_to_num(char *buffer) {
	int i;
	int templen;
	int sum = 0;
	templen = strlen(buffer);
	if (templen > 10) {
		return -1;
	}
	for (i = 0; i < templen; i++) {
		buffer[i] -= 0x30;
		if ((buffer[i] >= 0) && (buffer[i] <= 9)) {
		} else {
			return -1;
		}
	}
	int j = 0;
	for (i = 0; i < templen; i++) {
		sum += buffer[i] * num_tab[10 - templen + j];
		j++;
	}
	return sum;
}

//发送标准协议包数据
//包头结构和数据体必须是连续的存储
void Octopus_Send(STRUCT_Octopus *pOctopus) {
	//包头 + 数据 + CRC + 结束标记
	debug_printfBuff((uint8_t*) pOctopus, OCTOPUS_DATA_HEAD_SIZE + pOctopus->length + 2);
}
//包头和数据体必须是连续的存储
void Octopus_S_Send(STRUCT_Octopus_S *pOctopus) {
	debug_printfBuff((uint8_t*) pOctopus, OCTOPUS_S_DATA_HEAD_SIZE + pOctopus->length);
}

//发送非标准协议包数据,兼容YINENG
//八爪鱼上位机支持如下简易数据包
void Octopus_S_Send_Cmd(uint8_t cmd, uint8_t *buf, uint8_t len) {
	usart1_dma.send_buf[0] = COMM_HAND1;
	usart1_dma.send_buf[1] = COMM_HAND2;
	usart1_dma.send_buf[2] = cmd; //命令
	usart1_dma.send_buf[3] = (usart1_dma.send_buf[3] == 0xFE) ? 0xFE : 0; //索引
	usart1_dma.send_buf[4] = len + 6; //长度
	memcpy(&usart1_dma.send_buf[5], buf, len);
	usart1_dma.send_buf[usart1_dma.send_buf[4] - 1] = crc8(usart1_dma.send_buf, usart1_dma.send_buf[4] - 1);
	debug_printfBuff(usart1_dma.send_buf, usart1_dma.send_buf[4]);
}

//包头结构和数据体不是连续的存储，实际数据连续存储在pData
void OctopusPackagePrint(STRUCT_Octopus *pOctopus) {
	if (pOctopus->head == OCTOPUS_PROTOCOL_HEAD) {
		//Octopus_Send(pOctopus->pData);
		debug_printfBuff((uint8_t*) pOctopus->pData, OCTOPUS_DATA_HEAD_SIZE + pOctopus->length + 2);
	} else if (pOctopus->head == OCTOPUS_PROTOCOL_COMM_HEAD) {
		//Octopus_S_Send(pOctopus->pData);
		debug_printfBuff((uint8_t*) pOctopus->pData, OCTOPUS_S_DATA_HEAD_SIZE + pOctopus->length);
	} else {
		debug_printfBuff((uint8_t*) pOctopus, OCTOPUS_DATA_HEAD_SIZE + pOctopus->length + 2);
	}
}

void cmd_Test(int argc, char *argv[], uint8_t index) {
	int i;
	char str[][4] = {"abc","haha","no"};
	debug_printf("\n***************************************************\n");
	debug_printf("function:cmd_test%s\n");
	debug_printf("usage:%s\n", mv_shell_cmds[index].CmdUsage);
	for (i = 0; i < argc; i++) {
		if (i == 0) {
			debug_printf("%s\n", argv[i]);
		} else {
			debug_printf("%d\n", char_to_num(argv[i]));
		}
	}
}

void cmd_Displayhelp(int argc, char *argv[], uint8_t index) {
	int i = 0;
	debug_printf("\n***************************************************\nCommand list:\n");
	//Debug_printf("\tcmd\t\thelpinfo\t\n");
	//Debug_printf("\n");
	while (1) {
		if (mv_shell_cmds[i].Cmd == NULL)
			break;

		//Debug_printf("\t%s\t\t%s\t\n", mv_shell_cmds[i].Cmd, mv_shell_cmds[i].CmdUsage);
		debug_printf("\t%-10s%s\t\n", mv_shell_cmds[i].Cmd, mv_shell_cmds[i].CmdUsage);

		i++;
	}
	debug_printf("example:test 123 456\n");
	//Debug_printf("\n");
	//Debug_printf("***************************************************\n");
}

void cmd_GetAdcValue(int argc, char *argv[], uint8_t index) {
	//Debug_printfBuff((uint8_t*) ADC_Value, 8);
}

//////////////////////////////////////////////////////////////////////////////////////
//串口字节流命令处理函数，注意大小端存储循序的不同
/*
 OCCOMPROTOCAL_HEAD = $0101; //八爪鱼标准协议头
 OCCOMPROTOCAL_HEAD2 = $0AFF; //简易或短包协议头码
 //OCCOMPROTOCAL_HEAD2_BigEndian = $FF0A; //大端存储
 OCCOMPROTOCAL_END = $7E; //结束标记
 OCCOMPROTOCAL_ACK = "y"

 OCCOMPROTOCAL_DATA = $DA; //发送数据 包最大是512字节
 OCCOMPROTOCAL_DATA_COMPLETE = $DC; // 数据发送完成
 OCCOMPROTOCAL_OVER = OCCOMPROTOCAL_DATA_COMPLETE; // 任务结束标记

 OCCOMPROTOCAL_FLASH_READ = $F0;//读Flash
 OCCOMPROTOCAL_FLASH_WRITE = $F1;//写FLASH
 */
//////////////////////////////////////////////////////////////////////////////////////
void Octopus_Process(STRUCT_Octopus *pOctopus) {
	uint8_t i = 0;
	uint8_t temp[10];

	if (pOctopus->length > OCTOPUS_DATA_MAX_LENGTH) {
		debug_printf("Invalid data length = %d\n", pOctopus->length);
	}

	switch (pOctopus->pid) {
	case 0x00:
		//测试命令，原路返回
		OctopusPackagePrint(pOctopus);
		//for (i =0;i<1000;i++)
		//{//测试连续发送
		//	Debug_printf("Test:%d\r\n",i);
		//	if(i >254)break;
		//}
		break;
	case 0x01:		//测试应答
		Octopus_S_Send_Cmd(pOctopus->pid, (uint8_t*) ("Y"), 1);
		break;
	case 0x02:
		cmd_GetAdcValue(0, 0, 0);
		break;
	case 0x20: //读取ADC电流电压温度
		i = 0;
		/*temp[i++] = ADC_Value[0] / 65536 % 256;
		temp[i++] = ADC_Value[0] % 65536 / 256;
		temp[i++] = 0; //chan.upvolt%65536%256;

		temp[i++] = ADC_Value[1] / 65536 % 256;
		temp[i++] = ADC_Value[1] % 65536 / 256;
		temp[i++] = 0; //chan.upcurr%65536%256;

		temp[i++] = ADC_Value[2] / 256;
		temp[i++] = ADC_Value[2] % 256;*/

		Octopus_S_Send_Cmd(pOctopus->pid, temp, i);
		break;

	case 0xDA:		//传输数据包
		//printf("\n");
		break;
	case 0xDB:
		//printf("\n");
		break;
	case 0xDC:		//数据包传输完成 data complete
		/*
		 //接收文件长度
		 uint32_t applen = pOctopus->pData[7] * 256 + pOctopus->pData[8];
		 //接收文件的累加和
		 uint32_t appmod = pOctopus->pData[9] * 65536 * 256 + pOctopus->pData[10] * 65536 + pOctopus->pData[11] * 256 + pOctopus->pData[12];
		 //WriteFlash(&applen, &appmod);
		 Octopus_Send_Other_S(pOctopus->pData[2], &pOctopus->pData[2], 1, false);		//反馈给上位机
		 //asm("CPSID  I");
		 //关中断
		 //flash_restart_inapp();
		 */
		break;

	//写flash指令收到hex bin文件后的处理
	case 0xF1:
		/*
		 uint16_t len = pOctopus->pData[5] - 4;		//前4个字节为32位地址
		 uint32_t address = pOctopus->pData[7] * 65536 * 256 + pOctopus->pData[8] * 65536 + pOctopus->pData[9] * 256 + pOctopus->pData[10];
		 WriteFlash(address, len, &urbuf[11]);
		 Octopus_Send_Other_S(pOctopus->pData[2], &pOctopus->pData[7], 4, false);		//反馈给上位机
		 */
		break;
	default:
		debug_printf("No this command!\n");
		break;
	}
	//sprintf(buffer, "%d\n", *p);
	//Debug_printfBuff((uint8_t*) pOctopus, OCTOPUS_DATA_HEAD_SIZE);
	//Debug_printfBuff(pOctopus->bData, OCTOPUS_DATA_HEAD_SIZE + pOctopus->length + 2);
}

//////////////////////////////////////////////////////////////////////////////////////
//串口字符串命令处理函数
//////////////////////////////////////////////////////////////////////////////////////
int Command_Proess() {
	int i = 0;
	int j = 0;
	uint k = 0;
	int ii = 0;
	int argc = 0;
	char *argv[OCTOPUS_CMD_ARG_MAX_COUNT];
	char temp[OCTOPUS_ONE_ARG_MAX_LENGTH];
	uint8_t cmdBuffer[OCTOPUS_CMD_ARG_MAX_COUNT][OCTOPUS_ONE_ARG_MAX_LENGTH];

	while (1) {
		if (mv_shell_cmds[i].Cmd == NULL) {
			return j;
		}

		if (memcmp(usart1_dma.recv_buf, mv_shell_cmds[i].Cmd, strlen(mv_shell_cmds[i].Cmd)) == 0) {
			argc = 0;
			ii = 0;
			memset(temp, 0, sizeof(temp));
			for (k = 0; k < strlen((char*) usart1_dma.recv_buf); k++) {
				if (usart1_dma.recv_buf[k] == 0x20) //空格和回车
						{
					memcpy(cmdBuffer + argc, temp, sizeof(temp));
					memset(temp, 0, sizeof(temp));
					argv[argc] = (char*) (cmdBuffer + argc); //指向参数的位置
					ii = 0;
					argc++;
					if (argc == OCTOPUS_CMD_ARG_MAX_COUNT)
						break;
				} else if (usart1_dma.recv_buf[k] == 0x0D)
					break;
				else {
					temp[ii] = usart1_dma.recv_buf[k];
					ii++;
					if (argc >= OCTOPUS_ONE_ARG_MAX_LENGTH)
						ii = OCTOPUS_ONE_ARG_MAX_LENGTH;

				}
			}

			(mv_shell_cmds[i].CmdHandler)(argc, argv, i);
			j++;
		}
		i++;
	}
	return j;
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//串口数据解析函数,在main while中调用，或FreeRTOS的线程中调用
void mv_shell_task(void *param) {
	uint8_t cmdIndex = 0;
	STRUCT_Octopus octopus_data;
	//STRUCT_Octopus_S octopus_S_data;
	//Debug_printf("\n***************************************************\n");
	//Debug_printf("Octopus shell command line\n");
	//Displayhelp(0, param, 0);

	//while (1)
	{
		if (usart1_dma.recv_hander_status == 0)	//没有数据处理
			return;	//continue;

		if (usart1_dma.recv_length <= 0) {	//忽略空
			usart1_dma.recv_hander_status = 0;	//处理完毕
			return;	//continue;
		}
		//octopus_data.pData = (uint8_t　*)malloc(sizeof(uint8_t));
		//回显串口数据
		//Debug_printf4Buff(usart1_dma.recv_buf, usart1_dma.recv_len);
		////////////////////////////////////////////////////////////////////////
		//字符命令处理
		if (Command_Proess() > 0) {
			cmdIndex = 0;
			usart1_dma.recv_hander_status = 0;	//处理完毕
			return;	//continue;
		}
		////////////////////////////////////////////////////////////////////////
		//字节流命令处理
		STRUCT_Octopus *pOctopus = (STRUCT_Octopus*) (&usart1_dma.recv_buf[cmdIndex]);
		while (usart1_dma.recv_length >= 3) {
			//标准字节流协议
			if (pOctopus->head == OCTOPUS_PROTOCOL_HEAD && usart1_dma.recv_length >= OCTOPUS_DATA_HEAD_SIZE) {
				octopus_data.head = 0x0101;
				octopus_data.pid = MakeWord(usart1_dma.recv_buf[3], usart1_dma.recv_buf[2]);
				octopus_data.index = usart1_dma.recv_buf[4];
				octopus_data.length = MakeWord(usart1_dma.recv_buf[6], usart1_dma.recv_buf[5]);

				octopus_data.pData = &usart1_dma.recv_buf[cmdIndex];		//指向原始数据的位置
				Octopus_Process(&octopus_data);
				//usart1_dma.recv_hander_status = 0;	//处理完毕
				//cmdIndex = 0;
				break;
			}
			//短字节流协议 最短3个字节构成一个数据包
			else if (pOctopus->head == OCTOPUS_PROTOCOL_COMM_HEAD && usart1_dma.recv_length >= 3) {
				//转换成octopus标准字节流协议后再做处理
				octopus_data.head = 0x0AFF;				//第0、1个字节为头码
				octopus_data.pid = usart1_dma.recv_buf[2];				    //命令ID
				octopus_data.index = usart1_dma.recv_buf[3];				//索引或设备地址
				octopus_data.length = usart1_dma.recv_buf[4];				//有效数据长度
				octopus_data.pData = &usart1_dma.recv_buf[cmdIndex];				//指向原始数据的位置
				Octopus_Process(&octopus_data);
				//usart1_dma.recv_hander_status = 0;	//处理完毕
				//cmdIndex = 0;
				break;
			} else {
				cmdIndex++;
				if (cmdIndex > (usart1_dma.recv_length - OCTOPUS_DATA_HEAD_SIZE)) {
					cmdIndex = 0;
					break;	//数据不足
				}
				pOctopus = (STRUCT_Octopus*) (&usart1_dma.recv_buf[cmdIndex]);
			}
		}
		/////////////////////////////////////////////////////////////////////////
		pOctopus->head = 0;
		pOctopus->pid = 0;
		usart1_dma.recv_length = 0;
		usart1_dma.recv_hander_status = 0;	//处理完毕
		memset(usart1_dma.recv_buf,0,BUFFERSIZE);
		//到这里数据不足，忽略字节流
		//vTaskDelay(1);
	}
}
