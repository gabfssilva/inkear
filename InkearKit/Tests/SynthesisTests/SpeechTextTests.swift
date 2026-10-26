import Testing

@testable import Synthesis

@Suite struct SpeechTextTests {
  @Test(arguments: [
    ("Havia 1.000.000 de habitantes.", "Havia um milhão de habitantes."),
    ("Em 1984 houve uma greve.", "Em mil novecentos e oitenta e quatro houve uma greve."),
    ("Eram 3,5 toneladas.", "Eram três vírgula cinco toneladas."),
    ("Cresceu 45% no ano.", "Cresceu quarenta e cinco por cento no ano."),
    ("Custa R$ 10,50 hoje.", "Custa dez reais e cinquenta centavos hoje."),
    ("Custa R$ 1.", "Custa um real."),
    ("Custa US$ 3.", "Custa três dólares."),
    ("O 1º lugar.", "O primeiro lugar."),
    ("A 1ª vez.", "A primeira vez."),
    ("Foi o 21º colocado.", "Foi o vigésimo primeiro colocado."),
    ("Foi o 2.º dia.", "Foi o segundo dia."),
    ("Em 1º de maio.", "Em primeiro de maio."),
    ("Eram 1 página.", "Eram uma página."),
    ("Havia 2 horas.", "Havia duas horas."),
    ("Chegaram 200 pessoas.", "Chegaram duzentas pessoas."),
    ("Há 21 pessoas.", "Há vinte e uma pessoas."),
    ("Há 1 dia e 2 dias.", "Há um dia e dois dias."),
    ("Eram 1 mapa.", "Eram um mapa."),
    ("Foi em 1990 e 2 anos.", "Foi em mil novecentos e noventa e dois anos."),
    ("Veja 1990-1995.", "Veja mil novecentos e noventa a mil novecentos e noventa e cinco."),
    ("Às 14h30.", "Às catorze e trinta."),
    ("Às 14h.", "Às catorze horas."),
    ("Às 14:30.", "Às catorze e trinta."),
    ("Em 12/05/1990 houve.", "Em doze de maio de mil novecentos e noventa houve."),
    ("Sr. Silva chegou.", "Senhor Silva chegou."),
    ("Dr. Souza e Profa. Lima.", "Doutor Souza e Professora Lima."),
    ("Veja pág. 12.", "Veja página doze."),
    ("Veja págs. 12 e 13.", "Veja páginas doze e treze."),
    ("Ver cap. IV.", "Ver capítulo quatro."),
    ("Foi no século XIX.", "Foi no século dezenove."),
    ("Capítulo IV começa.", "Capítulo quatro começa."),
    ("Veja tab. 2.", "Veja tabela dois."),
    ("Foi em 300 a.C.", "Foi em trezentos antes de Cristo."),
    ("Pão & leite.", "Pão e leite."),
    ("Veja 2 + 2 = 4.", "Veja dois mais dois igual a quatro."),
    ("Lado 3 × 4.", "Lado três vezes quatro."),
    ("Veja o § 3.", "Veja o parágrafo três."),
    ("Está a 25°C hoje.", "Está a vinte e cinco graus Celsius hoje."),
    ("Segundo Silva[12] confirmou[3, 4].", "Segundo Silva confirmou."),
    ("Acesse https://www.exemplo.com.br/teste agora.", "Acesse exemplo ponto com ponto br agora."),
    ("Veja www.site.org.", "Veja site ponto org."),
  ])
  func portuguese(_ pair: (String, String)) {
    #expect(SpeechText.normalized(pair.0, language: "pt") == pair.1)
  }

  @Test(arguments: [
    ("Revenue grew 45% last year.", "Revenue grew forty-five percent last year."),
    ("It costs $10.50.", "It costs ten dollars and fifty cents."),
    ("It costs $1.", "It costs one dollar."),
    ("Costs $1,000.50 total.", "Costs one thousand dollars and fifty cents total."),
    ("Pi is 3.14.", "Pi is three point one four."),
    ("It has 1,000 units.", "It has one thousand units."),
    ("Stop at 1,000,000 people.", "Stop at one million people."),
    ("A 3.5% rise.", "A three point five percent rise."),
    ("In 1984 it began.", "In nineteen eighty-four it began."),
    ("In 1900 it ended.", "In nineteen hundred it ended."),
    ("In 1905 it ended.", "In nineteen oh five it ended."),
    ("In 2024 it ended.", "In twenty twenty-four it ended."),
    ("In 2005 it ended.", "In two thousand five it ended."),
    ("Between 1990-1995 there.", "Between nineteen ninety to nineteen ninety-five there."),
    ("The 1st chapter.", "The first chapter."),
    ("The 2nd and 3rd.", "The second and third."),
    ("The 21st century.", "The twenty-first century."),
    ("The 100th run.", "The one hundredth run."),
    ("Mr. Smith arrived.", "Mister Smith arrived."),
    ("Dr. Lee and Mrs. Kim.", "Doctor Lee and Missus Kim."),
    ("Ms. Ito wrote approx. 5 pages.", "Miz Ito wrote approximately five pages."),
    ("Prof. Smith.", "Professor Smith."),
    ("See Fig. 3.", "See Figure three."),
    ("Apples vs. oranges.", "Apples versus oranges."),
    ("Done, etc.", "Done, et cetera."),
    ("Chapter IV begins.", "Chapter four begins."),
    ("Part I is short.", "Part one is short."),
    ("Tom & Jerry.", "Tom and Jerry."),
    ("Read §3 now.", "Read section three now."),
    ("It is 25°C today.", "It is twenty-five degrees Celsius today."),
    ("Water boils at 2 + 2 = 4.", "Water boils at two plus two equals four."),
    ("Area is 3 × 4.", "Area is three times four."),
    ("See the book[12].", "See the book."),
    ("Visit https://www.example.org/docs now.", "Visit example dot org now."),
    ("Visit www.example.co.uk.", "Visit example dot co dot uk."),
    ("Read pp. 10 and 11.", "Read pages ten and eleven."),
    ("See p. 45.", "See page forty-five."),
    ("Ch. 3 is short.", "Chapter three is short."),
    ("No. 7 won.", "number seven won."),
    ("Sec. 2 applies.", "Section two applies."),
  ])
  func english(_ pair: (String, String)) {
    #expect(SpeechText.normalized(pair.0, language: "en") == pair.1)
  }

  @Test(arguments: [
    ("pt", "O rato roeu a roupa do rei de Roma."),
    ("en", "The quick brown fox jumps over the lazy dog."),
  ])
  func textWithoutNumbersIsUnchanged(_ pair: (String, String)) {
    #expect(SpeechText.normalized(pair.1, language: pair.0) == pair.1)
  }

  @Test func otherLanguagesGetOnlyTheLanguageIndependentRules() {
    #expect(SpeechText.normalized("Il a 3 chats[1].", language: "fr") == "Il a 3 chats.")
  }
}
