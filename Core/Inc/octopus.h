/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    octopus.h
 * @brief   This file contains all the function prototypes for
 *          the octopus.c file
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __OCTOPUS_H__
#define __OCTOPUS_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

/* USER CODE BEGIN Includes */
#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
/* USER CODE END Includes */

/* USER CODE BEGIN Private defines */
typedef void (*shell_cmd_func)(int32_t, char**, uint8_t index);

//variable length data types
//八爪鱼串口协议可变长度数据结构
#define OCTOPUS_VARIABLE_LENGTH_DATA_STRUCT

#ifdef OCTOPUS_VARIABLE_LENGTH_DATA_STRUCT
//Octopus八爪鱼串口协议数据结构(数据包的头部结构)
//包最小7个字节，整个传输数据的长度等于有效数据长度+7
typedef struct Octopus {
	uint16_t head; //数据包头部识别码 0x0101
	uint16_t pid; //数据包类型ID(命令)
	uint8_t index; //数据包索引
	uint16_t length; //有效数据的长度

	uint8_t *pData; //指向有效数据的位置或者实际需要操作的数据位置
    //uint8_t crc1;//用作校验
    //uint8_t crc2 || endFlag;//包结束标记为crc2 = 0x7E
} STRUCT_Octopus, *POctopus;

#else
//也可以定义成如下固定长度的数据结构
typedef struct Octopus{
	uint16_t head;//数据包头部识别码 0x0101
	uint16_t pid; //数据包类型ID
	uint8_t index;//数据包索引
	uint16_t length;//有效数据的长度

	uint8_t data[100];//最大100个字节数据
	uint8_t crc1;
	uint8_t crc2;//crc2 = 0x7E
} STRUCT_Octopus, *POctopus;
#endif

typedef struct Octopus_S {
	uint16_t head; //数据包头部识别码 0xFF0A
	uint8_t pid; //数据包类型ID(命令)
	uint8_t index; //数据包索引
	uint8_t length; //有效数据的长度

	uint8_t *pData; //指向有效数据的位置或者实际需要操作的数据位置
    //uint8_t crc1;//用作校验
    //uint8_t crc2 || endFlag;//包结束标记为crc2 = 0x7E
} STRUCT_Octopus_S, *POctopus_S;

typedef struct CLICmds_st {
	const char *Cmd;
	shell_cmd_func CmdHandler;
	const char *CmdUsage;
} STRUCT_Shell_Commands;

#define OCTOPUS_DATA_MAX_LENGTH 100
#define OCTOPUS_DATA_HEAD_SIZE 7
#define OCTOPUS_CMD_ARG_MAX_COUNT	3
#define OCTOPUS_ONE_ARG_MAX_LENGTH		10
#define OCTOPUS_PROTOCOL_HEAD 0x0101
#define OCTOPUS_PROTOCOL_COMM_HEAD 0x0AFF

//COMM 头码
#define COMM_HAND1      0xFF
#define COMM_HAND2      0x0A
#define OCTOPUS_S_DATA_HEAD_SIZE 5

/* USER CODE END Private defines */

void cmd_Test(int argc, char *argv[], uint8_t index);
void cmd_Displayhelp(int argc, char *argv[], uint8_t index);
void cmd_GetAdcValue(int argc, char *argv[], uint8_t index);

uint8_t crc8(uint8_t *buf, uint16_t len);
uint16_t MakeWord(uint8_t b1, uint8_t b0);
uint32_t char_to_num(char *buffer);



int Command_Proess();
void Octopus_Process(STRUCT_Octopus *pOctopus);

void Octopus_Send(STRUCT_Octopus *pOctopus);
void Octopus_S_Send_Cmd(uint8_t cmd, uint8_t *buf, uint8_t len);
void mv_shell_task(void *param);
/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __OCTOPUS_H__ */

