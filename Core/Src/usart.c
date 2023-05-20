/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    usart.c
 * @brief   This file provides code for the configuration
 *          of the USART instances.
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
/* Includes ------------------------------------------------------------------*/
#include "usart.h"
#include "string.h"
/* USER CODE BEGIN 0 */
STRUCT_USART_DMA usart1_dma;

//设备为RS485
#define USART1_RS485 1
#define USART_RS485_RESET   HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET);
#define USART_RS485_SET 	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_SET);
/* USER CODE END 0 */

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_rx;
DMA_HandleTypeDef hdma_usart1_tx;

/* USART1 init function */

void MX_USART1_UART_Init(void) {

	/* USER CODE BEGIN USART1_Init 0 */
#ifdef USART1_RS485
	USART_RS485_RESET
	;
#endif
	/* USER CODE END USART1_Init 0 */

	/* USER CODE BEGIN USART1_Init 1 */

	/* USER CODE END USART1_Init 1 */
	huart1.Instance = USART1;
	huart1.Init.BaudRate = 115200;
	huart1.Init.WordLength = UART_WORDLENGTH_8B;
	huart1.Init.StopBits = UART_STOPBITS_1;
	huart1.Init.Parity = UART_PARITY_NONE;
	huart1.Init.Mode = UART_MODE_TX_RX;
	huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
	huart1.Init.OverSampling = UART_OVERSAMPLING_16;
	huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
	huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
	huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
	if (HAL_UART_Init(&huart1) != HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK) {
		Error_Handler();
	}
	if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN USART1_Init 2 */
	__HAL_UART_ENABLE_IT(&huart1, UART_IT_IDLE);
	HAL_UART_Receive_DMA(&huart1, usart1_dma.dma_buf, BUFFERSIZE);
	/* USER CODE END USART1_Init 2 */

}

void HAL_UART_MspInit(UART_HandleTypeDef *uartHandle) {

	GPIO_InitTypeDef GPIO_InitStruct = { 0 };
	RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = { 0 };
	if (uartHandle->Instance == USART1) {
		/* USER CODE BEGIN USART1_MspInit 0 */

		/* USER CODE END USART1_MspInit 0 */

		/** Initializes the peripherals clock
		 */
		PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_USART1;
		PeriphClkInitStruct.Usart16ClockSelection = RCC_USART16CLKSOURCE_D2PCLK2;
		if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) {
			Error_Handler();
		}

		/* USART1 clock enable */
		__HAL_RCC_USART1_CLK_ENABLE();

		__HAL_RCC_GPIOB_CLK_ENABLE();
		/**USART1 GPIO Configuration
		 PB14     ------> USART1_TX
		 PB15     ------> USART1_RX
		 */
		GPIO_InitStruct.Pin = GPIO_PIN_14 | GPIO_PIN_15;
		GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
		GPIO_InitStruct.Pull = GPIO_NOPULL;
		GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
		GPIO_InitStruct.Alternate = GPIO_AF4_USART1;
		HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

		/* USART1 DMA Init */
		/* USART1_RX Init */
		hdma_usart1_rx.Instance = DMA1_Stream0;
		hdma_usart1_rx.Init.Request = DMA_REQUEST_USART1_RX;
		hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
		hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
		hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;
		hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
		hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
		hdma_usart1_rx.Init.Mode = DMA_NORMAL;
		hdma_usart1_rx.Init.Priority = DMA_PRIORITY_HIGH;
		hdma_usart1_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
		if (HAL_DMA_Init(&hdma_usart1_rx) != HAL_OK) {
			Error_Handler();
		}

		__HAL_LINKDMA(uartHandle, hdmarx, hdma_usart1_rx);

		/* USART1_TX Init */
		hdma_usart1_tx.Instance = DMA1_Stream1;
		hdma_usart1_tx.Init.Request = DMA_REQUEST_USART1_TX;
		hdma_usart1_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
		hdma_usart1_tx.Init.PeriphInc = DMA_PINC_DISABLE;
		hdma_usart1_tx.Init.MemInc = DMA_MINC_ENABLE;
		hdma_usart1_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
		hdma_usart1_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
		hdma_usart1_tx.Init.Mode = DMA_NORMAL;
		hdma_usart1_tx.Init.Priority = DMA_PRIORITY_HIGH;
		hdma_usart1_tx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
		if (HAL_DMA_Init(&hdma_usart1_tx) != HAL_OK) {
			Error_Handler();
		}

		__HAL_LINKDMA(uartHandle, hdmatx, hdma_usart1_tx);

		/* USART1 interrupt Init */
		HAL_NVIC_SetPriority(USART1_IRQn, 5, 0);
		HAL_NVIC_EnableIRQ(USART1_IRQn);
		/* USER CODE BEGIN USART1_MspInit 1 */

		/* USER CODE END USART1_MspInit 1 */
	}
}

void HAL_UART_MspDeInit(UART_HandleTypeDef *uartHandle) {

	if (uartHandle->Instance == USART1) {
		/* USER CODE BEGIN USART1_MspDeInit 0 */

		/* USER CODE END USART1_MspDeInit 0 */
		/* Peripheral clock disable */
		__HAL_RCC_USART1_CLK_DISABLE();

		/**USART1 GPIO Configuration
		 PB14     ------> USART1_TX
		 PB15     ------> USART1_RX
		 */
		HAL_GPIO_DeInit(GPIOB, GPIO_PIN_14 | GPIO_PIN_15);

		/* USART1 DMA DeInit */
		HAL_DMA_DeInit(uartHandle->hdmarx);
		HAL_DMA_DeInit(uartHandle->hdmatx);

		/* USART1 interrupt Deinit */
		HAL_NVIC_DisableIRQ(USART1_IRQn);
		/* USER CODE BEGIN USART1_MspDeInit 1 */

		/* USER CODE END USART1_MspDeInit 1 */
	}
}

/* USER CODE BEGIN 1 */

//阻塞
void USART1_Send(uint8_t *buffer, uint16_t length) {
	while (huart1.gState != HAL_UART_STATE_READY)
		;
	HAL_UART_Transmit(&huart1, buffer, length, 0xffff);
	//HAL_Delay(500);
	//while(USART_GetFlagStatus( USARTx, USART_FLAG_TC )	==	RESET);
}

//中断
void USART1_IT_Send(uint8_t *buffer, uint16_t length) {
	while (huart1.gState != HAL_UART_STATE_READY)
		;
	HAL_UART_Transmit_IT(&huart1, buffer, length);
	//HAL_Delay(500);
}
//DMA
void USART1_DMA_Send(uint8_t *buffer, uint16_t length) {
	while (HAL_DMA_GetState(&hdma_usart1_tx) != HAL_DMA_STATE_READY)
		;
	//__HAL_DMA_DISABLE(&hdma_usart1_tx);//关闭DMA
	HAL_UART_Transmit_DMA(&huart1, buffer, length);
	//while (HAL_DMA_GetState(&hdma_usart1_tx) != HAL_DMA_STATE_RESET)
	//		;
	HAL_Delay(30);
}
//////////////////////////////////////////////////////////////////////////////////
void debug_printf(const char *format, ...) {
	uint32_t length = 0;
	va_list args;
	va_start(args, format);
	memset(usart1_dma.send_buf, 0, BUFFERSIZE);
	length = vsnprintf((char*) usart1_dma.send_buf, sizeof(usart1_dma.send_buf), (char*) format, args);
#ifdef USART1_RS485
	USART_RS485_RESET;
#endif
	USART1_DMA_Send(usart1_dma.send_buf, length);
#ifdef USART1_RS485
	USART_RS485_SET;
#endif
}

void debug_printfBuff(uint8_t *buffer, uint16_t length) {
#ifdef USART1_RS485
	USART_RS485_RESET;
#endif
	USART1_DMA_Send(buffer, length);
#ifdef USART1_RS485
	USART_RS485_SET;
#endif
}

void Debug_printfRS485(const char *format, ...) {
	uint32_t length = 0;
	va_list args;
	//uartmd_send();
	va_start(args, format);
#ifdef USART1_RS485
	USART_RS485_RESET
	;
#endif
	memset(usart1_dma.send_buf, 0, BUFFERSIZE);
	length = vsnprintf((char*) usart1_dma.send_buf, sizeof(usart1_dma.send_buf), (char*) format, args);
	USART1_DMA_Send(usart1_dma.send_buf, length);
#ifdef USART1_RS485
	USART_RS485_SET
	;
#endif
	//uartmd_rece();
}

void Debug_printf4(const char *format, ...) {
	uint32_t length = 0;
	va_list args;
	va_start(args, format);
#ifdef USART1_RS485
	USART_RS485_RESET
	;
#endif
	memset(usart1_dma.send_buf, 0, BUFFERSIZE);
	length = vsnprintf((char*) usart1_dma.send_buf, sizeof(usart1_dma.send_buf), (char*) format, args);
	USART1_DMA_Send(usart1_dma.send_buf, length);
#ifdef USART1_RS485
	USART_RS485_SET
	;
#endif
}

void Debug_printf4Buff(uint8_t *buffer, uint16_t length) {
#ifdef USART1_RS485
	USART_RS485_SET
	;
#endif
	USART1_DMA_Send(buffer, length);
#ifdef USART1_RS485
	USART_RS485_SET
	;
#endif
}

//printf re directional
#ifdef __GNUC__
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif

PUTCHAR_PROTOTYPE {
	HAL_UART_Transmit(&huart1, (uint8_t*) &ch, 1, HAL_MAX_DELAY);
	return ch;
}

/* USER CODE END 1 */
