"""Teste de solução para o bug do env_nested_delimiter."""
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Solução 1: Usar model_config com from_attributes
class RiskConfig1(BaseModel):
    model_config = {"from_attributes": True}
    stop_loss_pct: float = Field(2.0, ge=0.1, le=20.0)
    take_profit_pct: float = Field(5.0, ge=0.1, le=50.0)
    max_drawdown_pct: float = Field(15.0, ge=1.0, le=50.0)
    max_open_trades: int = Field(3, ge=1, le=20)


class BotConfig1(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="_",
    )
    risk: RiskConfig1 = Field(default_factory=RiskConfig1)


# Solução 2: Sem default_factory, usar tipo direto
class RiskConfig2(BaseModel):
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 5.0
    max_drawdown_pct: float = 15.0
    max_open_trades: int = 3


class BotConfig2(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="_",
    )
    risk: RiskConfig2 = RiskConfig2()


# Solução 3: Sem default nenhum no campo
class BotConfig3(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="_",
    )
    risk: RiskConfig2  # sem default


print("=== Solução 1 (from_attributes) ===")
c1 = BotConfig1()
print(f"stop_loss={c1.risk.stop_loss_pct} take_profit={c1.risk.take_profit_pct} max_drawdown={c1.risk.max_drawdown_pct} max_open={c1.risk.max_open_trades}")

print("\n=== Solução 2 (default direto) ===")
c2 = BotConfig2()
print(f"stop_loss={c2.risk.stop_loss_pct} take_profit={c2.risk.take_profit_pct} max_drawdown={c2.risk.max_drawdown_pct} max_open={c2.risk.max_open_trades}")

print("\n=== Solução 3 (sem default) ===")
c3 = BotConfig3()
print(f"stop_loss={c3.risk.stop_loss_pct} take_profit={c3.risk.take_profit_pct} max_drawdown={c3.risk.max_drawdown_pct} max_open={c3.risk.max_open_trades}")
