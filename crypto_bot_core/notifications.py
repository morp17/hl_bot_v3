"""
Módulo de Notificações — Hyperliquid Production Bot v3.0
=========================================================
Suporte a Telegram, Discord e email para alertas de trading.

Requisitos:
- Type hints
- Tratamento de exceções com try/except
- Validação de inputs
- Logs estruturados
"""

from __future__ import annotations

import json
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests
from loguru import logger as log


# ──────────────────────────────────────────────
# Notificator
# ──────────────────────────────────────────────


class Notificator:
    """
    Sistema de notificações multicanal.

    Suporta:
    - Telegram (via bot API)
    - Discord (via webhook)
    - Email (via SMTP)
    """

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_pass: Optional[str] = None,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
    ) -> None:
        """
        Inicializa o notificador.

        Args:
            telegram_token: Token do bot Telegram.
            telegram_chat_id: Chat ID do Telegram.
            discord_webhook: URL do webhook do Discord.
            smtp_host: Host SMTP.
            smtp_port: Porta SMTP.
            smtp_user: Usuário SMTP.
            smtp_pass: Senha SMTP.
            email_from: Email remetente.
            email_to: Email destinatário.
        """
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.email_from = email_from
        self.email_to = email_to

    def send_telegram(self, message: str) -> bool:
        """
        Envia mensagem via Telegram.

        Args:
            message: Texto da mensagem.

        Returns:
            True se enviada com sucesso.
        """
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                log.debug("[TELEGRAM] Não configurado")
                return False

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }

            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()

            log.info("[TELEGRAM] Mensagem enviada")
            return True

        except requests.RequestException as e:
            log.error(f"[TELEGRAM] Erro ao enviar: {e}")
            return False
        except Exception as e:
            log.error(f"[TELEGRAM] Erro inesperado: {e}")
            return False

    def send_discord(self, message: str) -> bool:
        """
        Envia mensagem via Discord webhook.

        Args:
            message: Texto da mensagem.

        Returns:
            True se enviada com sucesso.
        """
        try:
            if not self.discord_webhook:
                log.debug("[DISCORD] Não configurado")
                return False

            payload = {"content": message}
            resp = requests.post(self.discord_webhook, json=payload, timeout=10)
            resp.raise_for_status()

            log.info("[DISCORD] Mensagem enviada")
            return True

        except requests.RequestException as e:
            log.error(f"[DISCORD] Erro ao enviar: {e}")
            return False
        except Exception as e:
            log.error(f"[DISCORD] Erro inesperado: {e}")
            return False

    def send_email(self, subject: str, body: str) -> bool:
        """
        Envia email via SMTP.

        Args:
            subject: Assunto do email.
            body: Corpo do email.

        Returns:
            True se enviado com sucesso.
        """
        try:
            if not all([self.smtp_host, self.smtp_user, self.smtp_pass, self.email_from, self.email_to]):
                log.debug("[EMAIL] Não configurado")
                return False

            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self.email_from
            msg["To"] = self.email_to

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)

            log.info(f"[EMAIL] Email enviado: {subject}")
            return True

        except smtplib.SMTPException as e:
            log.error(f"[EMAIL] Erro SMTP: {e}")
            return False
        except Exception as e:
            log.error(f"[EMAIL] Erro inesperado: {e}")
            return False

    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        pnl: Optional[float] = None,
    ) -> None:
        """
        Envia alerta de trade para todos os canais configurados.

        Args:
            action: "open" ou "close".
            symbol: Símbolo negociado.
            side: "buy" ou "sell".
            qty: Quantidade.
            price: Preço.
            pnl: PnL (opcional, apenas para fechamento).
        """
        try:
            emoji = "🟢" if side == "buy" else "🔴"
            action_text = "ABERTURA" if action == "open" else "FECHAMENTO"

            message = (
                f"{emoji} <b>Hyperliquid Bot</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"<b>Ação:</b> {action_text}\n"
                f"<b>Símbolo:</b> {symbol}\n"
                f"<b>Lado:</b> {side.upper()}\n"
                f"<b>Qty:</b> {qty}\n"
                f"<b>Preço:</b> ${price:,.2f}\n"
            )

            if pnl is not None:
                pnl_emoji = "✅" if pnl >= 0 else "❌"
                message += f"<b>PnL:</b> {pnl_emoji} ${pnl:+,.2f}\n"

            message += "━━━━━━━━━━━━━━━"

            self.send_telegram(message)
            self.send_discord(message)

        except Exception as e:
            log.error(f"[ALERT] Erro ao enviar alerta: {e}")

    def send_error(self, error_msg: str, context: str = "") -> None:
        """
        Envia alerta de erro.

        Args:
            error_msg: Mensagem de erro.
            context: Contexto do erro.
        """
        try:
            message = (
                f"🚨 <b>ERRO - Hyperliquid Bot</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"<b>Contexto:</b> {context}\n"
                f"<b>Erro:</b> {error_msg}\n"
                f"━━━━━━━━━━━━━━━"
            )

            self.send_telegram(message)
            self.send_discord(message)

            if self.email_to:
                self.send_email(
                    subject=f"[HYPERBOT] Erro: {context}",
                    body=f"Contexto: {context}\nErro: {error_msg}",
                )

        except Exception as e:
            log.error(f"[ALERT] Erro ao enviar alerta de erro: {e}")
