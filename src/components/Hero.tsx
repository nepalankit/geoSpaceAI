
import { Button } from "@/components/ui/button";

const Hero = () => {
  return (
    <section className="relative py-20 md:py-32 overflow-hidden">
      {/* Background pattern */}
      <div className="absolute top-0 left-0 w-full h-full opacity-10">
        <div className="absolute top-0 right-0 w-1/2 h-1/2 bg-primary/10 rounded-full blur-3xl -translate-y-1/2"></div>
        <div className="absolute bottom-0 left-0 w-1/2 h-1/2 bg-secondary/10 rounded-full blur-3xl translate-y-1/2"></div>
      </div>
      
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center text-center space-y-8 md:space-y-12">
          <div className="space-y-4 max-w-[42rem]">
            <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold tracking-tighter">
              AI-Powered Environmental 
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary"> Anomaly Detection</span>
            </h1>
            <p className="text-muted-foreground md:text-xl max-w-[600px] mx-auto">
              GeoScopeAI uses advanced machine learning to detect wildfires, floods, illegal deforestation, 
              and land degradation from satellite and drone imagery.
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4">
            <Button size="lg" className="text-base">
              Get Started
            </Button>
            <Button size="lg" variant="outline" className="text-base">
              Learn More
            </Button>
          </div>
          
          <div className="w-full max-w-4xl mx-auto relative mt-8 md:mt-16">
            {/* World map visualization with anomaly markers */}
            <div className="relative rounded-lg overflow-hidden shadow-xl">
              <div className="bg-[#E6F2F5] dark:bg-[#102030] aspect-[16/9] rounded-lg">
                <svg viewBox="0 0 1200 600" className="w-full h-full">
                  {/* World map silhouette - simplified path */}
                  <path 
                    d="M209,91L229,118L269,113L270,123L302,136L315,136L318,152L304,166L317,191L300,216L273,207L257,226L277,253L270,281L255,270L236,270L241,310L253,341L262,350L244,360L241,380L254,380L260,388L236,402L239,419L228,416L213,413L204,427L209,438L193,447L187,455L206,481L208,493L189,493L191,504L202,513L205,535L164,544L151,547L138,536L132,549L122,553L155,580L181,582L186,565L202,550L224,545L237,531L247,532L280,539L286,552L287,562L276,580L241,587L223,577L225,566L208,557L190,559L176,551L166,566L152,560L133,563L123,562L67,508L72,487L94,487L109,482L83,462L76,449L81,434L53,411L39,394L42,379L66,372L64,354L57,345L70,329L82,329L100,343L109,359L132,362L143,343L136,324L121,306L140,303L149,287L134,277L132,258L151,251L156,235L173,229L185,212L194,192L193,176L160,145L150,128L155,104L138,95L128,98L121,103L118,80L137,72L183,94L209,91M456,103L479,120L496,101L500,75L517,58L530,76L534,101L553,108L585,105L566,126L568,147L595,147L613,155L614,172L638,172L638,194L608,231L595,232L576,212L540,209L526,235L539,256L572,256L574,273L600,286L592,330L580,328L565,348L570,364L549,378L537,354L499,362L481,352L481,337L492,323L495,304L509,304L524,283L500,280L490,273L503,254L499,235L481,235L471,227L489,210L478,191L458,206L431,203L409,175L399,190L397,215L417,228L397,243L380,244L378,255L400,270L409,296L400,306L385,307L370,316L330,338L309,344L302,321L301,290L288,267L289,241L310,225L317,198L343,163L358,135L374,156L381,182L416,182L416,162L421,140L440,104L456,103M723,138L765,147L774,172L756,188L713,195L736,204L732,217L709,219L714,233L741,236L758,228L766,213L778,215L794,202L789,229L773,250L759,259L760,271L781,279L788,295L749,309L755,325L779,342L750,347L732,357L710,361L703,341L683,314L677,269L676,248L694,242L702,222L694,216L690,207L693,184L677,178L669,160L684,152L723,138M784,169L725,165L710,176L729,171L784,169M880,191L853,205L845,216L822,216L819,233L823,255L814,260L805,279L791,273L778,288L761,307L765,319L741,336L734,365L714,365L712,384L730,398L725,419L734,428L721,440L738,457L764,456L806,464L833,450L834,432L855,424L862,408L896,417L913,407L920,420L905,450L920,476L940,476L950,462L972,459L983,468L996,456L1008,464L1018,460L1007,443L1014,423L1005,400L1025,392L1030,379L1056,365L1100,366L1107,354L1121,348L1148,344L1156,331L1188,333L1195,340L1162,380L1130,404L1128,429L1110,437L1090,452L1069,450L1027,481L1024,500L1041,520L1071,525L1091,543L1083,550L1089,565L1107,571L1126,564L1130,540L1155,509L1170,491L1200,483L1200,478L1181,478L1180,460L1190,455L1200,460L1200,385L1183,380L1168,389L1151,373L1139,379L1136,390L1115,384L1105,371L1115,355L1105,338L1121,328L1139,335L1149,324L1155,300L1179,284L1200,281L1200,247L1193,241L1185,257L1173,257L1169,242L1182,230L1197,206L1200,187L1200,112L1189,87L1188,60L1174,64L1154,87L1127,99L1097,114L1089,128L1059,140L1051,132L1041,147L1030,154L1023,170L1008,163L1004,178L983,179L983,162L996,155L1002,142L1018,125L1019,98L988,94L985,104L940,141L926,139L916,145L891,151L876,166L880,191M857,209L851,188L874,179L883,198L880,211L857,209M895,246L887,257L901,263L898,276L887,280L909,295L924,290L896,313L883,307L876,313L862,309L860,319L871,336L858,337L855,328L840,340L823,334L811,341L795,333L802,325L822,320L826,302L839,283L848,302L856,303L866,296L875,294L886,279L895,283L912,278L912,262L895,246"
                    fill="#A3CBD9"
                    stroke="#4A8C9F"
                    strokeWidth="2"
                    className="dark:fill-[#1D4559] dark:stroke-[#2D5E77]"
                  />
                  
                  {/* Anomaly points with pulsing effect */}
                  {/* Wildfire in Americas */}
                  <circle cx="250" cy="250" r="8" className="fill-fire-600 animate-pulse-warning" style={{ animationDuration: '3s' }} />
                  <circle cx="250" cy="250" r="12" className="fill-none stroke-fire-600 stroke-2 opacity-50 animate-ping" style={{ animationDuration: '3s' }} />
                  
                  {/* Flood in Asia */}
                  <circle cx="800" cy="300" r="8" className="fill-water-600 animate-pulse-warning" style={{ animationDuration: '4s' }} />
                  <circle cx="800" cy="300" r="12" className="fill-none stroke-water-600 stroke-2 opacity-50 animate-ping" style={{ animationDuration: '4s' }} />
                  
                  {/* Deforestation in Amazon */}
                  <circle cx="350" cy="400" r="8" className="fill-forest-600 animate-pulse-warning" style={{ animationDuration: '3.5s' }} />
                  <circle cx="350" cy="400" r="12" className="fill-none stroke-forest-600 stroke-2 opacity-50 animate-ping" style={{ animationDuration: '3.5s' }} />
                  
                  {/* Land degradation in Africa */}
                  <circle cx="550" cy="350" r="8" className="fill-earth-600 animate-pulse-warning" style={{ animationDuration: '3.8s' }} />
                  <circle cx="550" cy="350" r="12" className="fill-none stroke-earth-600 stroke-2 opacity-50 animate-ping" style={{ animationDuration: '3.8s' }} />
                </svg>
              </div>
              
              {/* Overlay gradient */}
              <div className="absolute bottom-0 left-0 right-0 h-1/3 bg-gradient-to-t from-background/90 to-transparent"></div>
              
              {/* Caption */}
              <div className="absolute bottom-0 left-0 w-full p-4 md:p-6 text-left">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-background/80 backdrop-blur-sm text-sm font-medium mb-2">
                  <span className="w-2 h-2 rounded-full bg-fire-600 animate-pulse-warning"></span>
                  <span>Active Detection in Progress</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex flex-wrap justify-center gap-x-8 gap-y-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-fire-600 rounded-full"></div>
              <span>Wildfire Detection</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-water-600 rounded-full"></div>
              <span>Flood Monitoring</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-forest-600 rounded-full"></div>
              <span>Deforestation Tracking</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-earth-600 rounded-full"></div>
              <span>Land Degradation Analysis</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
