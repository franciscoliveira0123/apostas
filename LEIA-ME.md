# Análise de Apostas — versão web

Página com 4 módulos (Jogos, Tapete, Registo, Desempenho), atualizada automaticamente por um robô gratuito do GitHub.

## Como funciona

1. Duas vezes por dia (mais um reforço à sexta e ao sábado), o GitHub corre o `atualizar.py`.
2. O robô descarrega os ficheiros do football-data.co.uk, calcula os ratings das equipas e grava `dados.json` e `backtest.json`.
3. A página (`index.html`) é publicada com esses ficheiros no GitHub Pages. Ao abrir, já está atualizada.

## Modelo (v2)

- **Ratings ajustados ao adversário** (modelo de Maher/Dixon-Coles por máxima verosimilhança), com vantagem casa de cada liga, ponderação temporal (`meia_vida_dias`) e regularização (`regularizacao_k`).
- **Remates à baliza** misturados com os golos (`peso_golos`): são mais estáveis e preveem melhor o futuro do que os golos.
- **Prior para promovidas e despromovidas** no início de cada época.
- **ρ de Dixon-Coles e α (peso do modelo face ao mercado) estimados automaticamente** com épocas anteriores (fora da amostra). Se o modelo não acrescentar informação, o α fica próximo de 0 e a probabilidade final aproxima-se da do mercado.
- **Validação por liga:** a liga só é "validada" se o modelo combinado bater o mercado por mais de 2 erros-padrão, com CLV positivo e amostra suficiente.

## Sinais na página

| Sinal | Significado | Stake sugerido |
|---|---|---|
| **Valor de preço** | A tua odd está acima do preço justo da Pinnacle/Betfair (sem margem). Sinal mais robusto. | Kelly fracionado |
| **Valor do modelo** | O modelo vê valor **e** a liga está validada. | Kelly fracionado |
| **A observar** | O modelo vê valor mas a liga não está validada. | 0 (só registo) |
| **Rever** | O modelo diverge muito do mercado: falta informação (lesões, rotação). | 0 |

O "valor de preço" depende da **odd real da tua casa**: introduz-a no detalhe do jogo ("A minha odd").

## Configuração inicial (uma vez)

1. **Criar o repositório.** No GitHub: botão **+** (canto superior direito) › **New repository**. Nome: `apostas`. Visibilidade: **Public** (obrigatório para o GitHub Pages gratuito). Marca **Add a README file**. **Create repository**.
2. **Enviar os ficheiros.** No repositório: **Add file › Upload files**. Arrasta `index.html`, `atualizar.py`, `config.json` e `LEIA-ME.md`. **Commit changes**.
3. **Criar o robô.** **Add file › Create new file**. No nome escreve exatamente `.github/workflows/atualizar.yml` (as barras criam as pastas). Cola o conteúdo do ficheiro `atualizar.yml`. **Commit changes**.
4. **Ativar o GitHub Pages.** **Settings › Pages**. Em *Build and deployment › Source*, escolhe **GitHub Actions**.
5. **Primeira execução.** Separador **Actions** › **Atualizar dados e publicar** (à esquerda) › **Run workflow** › **Run workflow**. Demora 1–3 minutos.
6. **Abrir a página.** Endereço: `https://<o-teu-utilizador>.github.io/apostas/`. Também aparece em **Settings › Pages**. No telemóvel: menu do browser › **Adicionar ao ecrã principal**.

## Utilização semanal

- **Jogos:** filtra por dia, liga ou “Só com valor”. Toca num jogo para ver o detalhe e introduzir a odd da tua casa (o EV é recalculado). “+ Jogo manual” serve para taças e jogos em falta.
- **Tapete:** junta as pernas, lê o diagnóstico e o teste de stress, regista.
- **Registo:** as apostas das ligas importadas liquidam-se sozinhas quando o robô traz o resultado (com CLV). Jogos manuais: marcar Ganha/Perdida/Nula à mão.
- **Desempenho:** indicadores das tuas apostas e backtest por liga e época.
- **Botão Atualizar** (no topo): abre a página do robô no GitHub para forçar uma atualização (**Run workflow**).

## Notas importantes

- **O registo fica guardado no browser onde o fazes.** Não sincroniza entre PC e telemóvel. Usa **Registo › Exportar cópia** com regularidade (e **Importar** para passar de um dispositivo para outro). Limpar os dados do browser apaga o registo.
- **Repositório público:** o código e os dados de futebol ficam visíveis; as tuas apostas não (estão só no browser).
- **Inatividade:** o GitHub suspende as tarefas agendadas de repositórios públicos sem alterações durante 60 dias. Se a página deixar de atualizar, vai a **Actions** e volta a ativar o workflow (ou faz um commit qualquer).
- **Próximos jogos:** dependem do `fixtures.csv` do football-data.co.uk, normalmente publicado perto da jornada. A meio da semana é normal aparecerem poucos ou nenhuns jogos.
- **Ligas e parâmetros do robô:** edita `config.json` no GitHub (lápis ✏️). `"ativa": true/false` liga ou desliga uma liga; `meia_vida_dias` e `regularizacao_k` mudam o modelo. Ao gravar, a página é republicada automaticamente.

## Se algo falhar

- **Página em branco ou “Sem dados”:** confirma em **Actions** que a última execução está verde. Se estiver vermelha, abre-a e copia a mensagem de erro.
- **Erro no passo “Deploy”:** confirma que em **Settings › Pages** a fonte é **GitHub Actions**.
- **Erro “Descarregar dados”:** o football-data.co.uk pode ter estado indisponível; a página anterior mantém-se e a próxima execução tenta de novo.

## Jogo responsável

Usar apenas operadores licenciados pelo SRIJ. Fazer primeiro a fase de teste sem dinheiro (registar com stake 0) até haver uma amostra com CLV consistente.
