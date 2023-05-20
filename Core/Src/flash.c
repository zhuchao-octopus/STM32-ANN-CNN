/*
 * flash.c
 *
 *  Created on: Mar 10, 2023
 *      Author: lenovo
 */

/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "usart.h"

#include "flash.h"

#define FLASH_PAGE_LEN      0x400
#define FLASH_USIZE         128

#define FLASH_ADD           0x8000000
#define FLASH_BOOTLOAD_SIZE 0x2000 //0x1800
#define FLASH_APP_ADD       0x8000000+FLASH_BOOTLOAD_SIZE

#define FLASH_PEAK_ADD      0x8008000-0xC00
#define FLASH_SYS_ADD       0x8008000-0x800
#define FLASH_USER_ADD      0x8008000-0x400

//系统数据存放在 0x8008000处 最大 1024 字节
#define FLASH_SYS_BOOT     0        //2字节-启动区域标记
#define FLASH_SYS_ALEN     2        //2字节-应用程序长度
#define FLASH_SYS_AMOD     4        //4字节-应用程序32位和检验
#define FLASH_SYS_LEDCATH  9        //1字节-校准值（斜率与失调）
#define FLASH_SYS_CALIB    10       //32字节-校准值（斜率与失调）
#define FLASH_SYS_MESS     50       //40字节-设备信息

//用户数据存放在 0x800B000处 最大 1024 字节
#define FLASH_USER_PROT    0        //14字节-保护参数:电压,电流,功率,温度
#define FLASH_USER_OTHER   20       //10字节-其他参数:风扇启温,停温,通道线阻,均标
#define FLASH_USER_CHAN    30       //34字节-通道参数


/* USER CODE BEGIN 4 */

/*FLASH写入程序*/
void WriteFlash(uint32_t L, uint32_t Data[], uint32_t addr) {

}

/*FLASH读取打印程序*/
void PrintFlashTest(uint32_t L, uint32_t addr) {
	uint32_t i = 0;
	for (i = 0; i < L; i++) {
		//printf("\naddr is:0x%x, data is:0x%x", addr + i * 4, *(__IO uint32_t*) (addr + i * 4));
	}
}

/* USER CODE END 4 */

/***********************************************************
 * 功能: 系统数据读取
 * 输入: 地址，长度，缓冲
 * 返回: 无
 ***********************************************************/
void Flash_Read(uint16_t add, uint16_t len, void *buf) {
	uint8_t *dst = (uint8_t*) buf;
	uint8_t *prt = (uint8_t*) (FLASH_SYS_ADD + add);
	while (len--) {
		*dst++ = *prt++;
	}
}




