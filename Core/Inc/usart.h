/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    usart.h
  * @brief   This file contains all the function prototypes for
  *          the usart.c file
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
#ifndef __USART_H__
#define __USART_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* USER CODE BEGIN Includes */
#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
/* USER CODE END Includes */

extern UART_HandleTypeDef huart1;

/* USER CODE BEGIN Private defines */

#define BUFFERSIZE 		100

typedef struct _USART_DMA_ {
	bool 	 dma_transmit_status;//dma传输完成标记�??1：传输完�??
	uint8_t  dma_buf[BUFFERSIZE];
	uint8_t  dma_length;//DMA传输长度

	bool 	 recv_hander_status;//接收处理标记�??1：有数据处理
	uint8_t  recv_length;
	uint8_t  recv_buf[BUFFERSIZE];
	uint8_t  send_buf[BUFFERSIZE];
} STRUCT_USART_DMA;

extern UART_HandleTypeDef huart1;
extern STRUCT_USART_DMA usart1_dma;
/* USER CODE END Private defines */

void MX_USART1_UART_Init(void);

/* USER CODE BEGIN Prototypes */
void USART1_DMA_Send(uint8_t *buffer, uint16_t length);

void debug_printf(const char *format, ...);
void debug_printfBuff(uint8_t *buffer, uint16_t length);
void Debug_printfRS485(const char *format, ...);
void Debug_printf4(const char *format, ...);
void Debug_printf4Buff(uint8_t *buffer, uint16_t length);

/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __USART_H__ */

