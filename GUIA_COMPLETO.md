# Guia Completo - Classificadores e Regressores

## 📚 Fontes de Dados

### OASIS (`oasis_longitudinal_demographic.csv`)
- **Usado em**: Classificação (SVM + DenseNet Classificação)
- **Contém**: Dados demográficos, clínicos (EDUC, MMSE, eTIV, nWBV, ASF, CDR)
- **Features**: Textura + Clínicas OASIS
- **Dimensão**: ~80 features

### PLANILHA (`planilha.csv`)
- **Usado em**: Regressão (XGBoost + DenseNet Regressão)
- **Contém**: Features morfológicas pré-calculadas (area, perimeter, circularity, eccentricity, solidity, extent, mean_intensity)
- **Features**: Textura + Morfológicas + Clínicas OASIS
- **Dimensão**: ~87 features

## 🎯 Fluxo Completo

### Passo 1: Preparar Dados
```
Dados → Preparar Dados (80/20)
```
- Carrega ambos CSVs
- Split 80/20 por paciente estratificado
- Validação 20% do treino

### Passo 2a: Classificação com SVM
```
Dados → Extrair Características (Classificação)
SVM → Treinar SVM
SVM → Avaliar SVM (Acurácia, Sensibilidade, Especificidade)
SVM → Matriz de Confusão
```
**Features**: Textura + Clínicas OASIS (sem morfológicas)

### Passo 2b: Regressão com XGBoost
```
Dados → Extrair Características (Regressão)
XGBoost → Treinar XGBoost
XGBoost → Avaliar XGBoost (MAE, RMSE, R², gráficos)
XGBoost → Análise Temporal (idades crescem com visitas?)
```
**Features**: Textura + **Morfológicas PLANILHA** + Clínicas OASIS

### Passo 3: Classificação Profunda
```
DenseNet → Treinar Classificação (fine-tuning, curvas automáticas)
DenseNet → Avaliar Classificação
DenseNet → Curvas Classificação
```
**Entrada**: Imagens RGB 224×224

### Passo 4: Regressão Profunda
```
DenseNet → Treinar Regressão (curvas automáticas)
DenseNet → Avaliar Regressão (MAE, RMSE, R², dispersão)
```
**Entrada**: Imagens RGB 224×224

## 📊 Tabela Comparativa

| Modelo | CSV | Features Morfológicas | Entrada | Saída |
|--------|-----|----------------------|---------|-------|
| **SVM** | OASIS | ❌ Não | Textura + Clínicas | Demented/NonDemented |
| **XGBoost** | PLANILHA | ✅ Sim | Textura + Morfológicas + Clínicas | Idade (anos) |
| **DenseNet Classif** | OASIS | N/A | Imagens RGB | Demented/NonDemented |
| **DenseNet Regress** | OASIS | N/A | Imagens RGB | Idade (anos) |

## ✅ Requisitos Atendidos

### Regressores Implementados
1. ✅ **XGBoost (raso)**: Usa features morfológicas da planilha + textura + clínicas
2. ✅ **DenseNet (profundo)**: Usa imagens originais

### Suficiência das Entradas
**XGBoost**: ✅ SIM
- Morfológicas (forma cerebral) + Textura (densidade) + Clínicas (volume) = combinação robusta
- Esperado: MAE < 6 anos, R² > 0.65

**DenseNet**: ✅ SIM (com limitações)
- CNN aprende features automáticas
- Dataset pequeno (~300 exames) pode limitar generalização
- Esperado: MAE < 8 anos, R² > 0.50

### Análise Temporal
✅ **Implementado**: `analise_temporal()`
- Verifica se idades preditas em visitas posteriores ≥ anteriores
- Tolerância: -0.5 anos (ruído)
- Gráfico de trajetórias por paciente
- Percentual de pacientes com crescimento consistente

**Interpretação**:
- >70%: Modelo captura progressão temporal ✅
- <50%: Features inadequadas ou overfitting ❌

## 🔑 Diferenças Críticas

### Variáveis Separadas
- **Classificação**: `self.x_train_features` (sem morfológicas)
- **Regressão**: `self.x_train_features_reg` (COM morfológicas)

### Métodos Separados
- **Classificação**: `extrair_caracteristicas()`
- **Regressão**: `extrair_caracteristicas_regressao()`

### Funções de Extração
```python
# Morfológicas (planilha.csv)
extract_morphological_features(planilha_df, mri_ids)
# Retorna: [area, perimeter, circularity, eccentricity, solidity, extent, mean_intensity]

# Clínicas (oasis)
extract_clinical_features(oasis_df, mri_ids)
# Retorna: [EDUC, MMSE, eTIV, nWBV, ASF, Visit, Years_since_first, CDR]

# Textura (imagens)
extract_features(images)
# Retorna: GLCM multi-distância + stats + quadrantes (72 features)
```

## 🐛 Troubleshooting

### Erro: "Extraia as características de REGRESSÃO primeiro!"
**Solução**: Dados → Extrair Características (Regressão)

### Erro: MRI ID não encontrado na planilha
**Solução**: Defaults automáticos aplicados (OK)

### XGBoost MAE alto (>10 anos)
**Causas possíveis**:
- Features de classificação usadas (sem morfológicas)
- Dados não preparados corretamente
- Planilha incompleta

**Verificar**:
1. Menu correto: "Extrair Características (Regressão)"
2. Dimensão features: ~87 (não ~80)

## 📈 Métricas Esperadas

### XGBoost
- Excelente: MAE < 4 anos, R² > 0.75
- Bom: MAE < 6 anos, R² > 0.65
- Aceitável: MAE < 8 anos, R² > 0.50

### DenseNet Regressão
- Bom: MAE < 6 anos, R² > 0.60
- Aceitável: MAE < 9 anos, R² > 0.40

### Análise Temporal
- Excelente: >80% pacientes consistentes
- Bom: >70%
- Aceitável: >60%

## 💡 Notas Finais

1. **Classificação NÃO usa planilha**: Apenas OASIS
2. **Regressão OBRIGATORIAMENTE usa planilha**: Features morfológicas essenciais
3. **Sempre executar na ordem**: Preparar Dados → Extrair Features → Treinar → Avaliar
4. **Análise temporal** responde "idades crescem?" usando XGBoost com features morfológicas
