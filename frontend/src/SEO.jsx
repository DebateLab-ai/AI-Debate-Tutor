import { Helmet } from 'react-helmet-async'

const BASE_URL = 'https://debatelab.ai'

export function SEO({
  title = 'DebateLab - AI-Powered Debate Practice Platform',
  description = 'Practice debating with an AI opponent. Get instant feedback, improve your argumentation skills, and master debate techniques with personalized coaching.',
  keywords = 'debate, debate practice, AI debate, argumentation, debate training, debate skills, public speaking, debate coaching',
  image = `${BASE_URL}/og-image.png`,
  url = BASE_URL,
  type = 'website',
  noindex = false,
}) {
  const fullTitle = title.includes('DebateLab') ? title : `${title} | DebateLab`
  
  return (
    <Helmet>
      {/* Primary Meta Tags */}
      <title>{fullTitle}</title>
      <meta name="title" content={fullTitle} />
      <meta name="description" content={description} />
      <meta name="keywords" content={keywords} />
      {noindex && <meta name="robots" content="noindex, nofollow" />}
      
      {/* Open Graph / Facebook */}
      <meta property="og:type" content={type} />
      <meta property="og:url" content={url} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:image" content={image} />
      <meta property="og:image:width" content="1200" />
      <meta property="og:image:height" content="630" />
      <meta property="og:site_name" content="DebateLab" />
      
      {/* Twitter */}
      <meta property="twitter:card" content="summary_large_image" />
      <meta property="twitter:url" content={url} />
      <meta property="twitter:title" content={fullTitle} />
      <meta property="twitter:description" content={description} />
      <meta property="twitter:image" content={image} />
      
      {/* Canonical URL */}
      <link rel="canonical" href={url} />
      
      {/* Additional SEO Tags */}
      <meta name="geo.region" content="US" />
      <meta name="geo.placename" content="United States" />
      <meta name="language" content="English" />
      <meta name="revisit-after" content="7 days" />
      <meta name="distribution" content="global" />
      <meta name="rating" content="general" />
    </Helmet>
  )
}

export function StructuredData({ data }) {
  if (!data) return null
  
  return (
    <Helmet>
      <script type="application/ld+json">
        {JSON.stringify(data)}
      </script>
    </Helmet>
  )
}

// Predefined structured data for different pages
export const organizationSchema = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  name: 'DebateLab',
  url: BASE_URL,
  logo: `${BASE_URL}/favicon.png`,
  description: 'AI-powered debate practice platform for improving argumentation skills',
  sameAs: [
    'https://discord.gg/your-server-invite',
  ],
}

export const websiteSchema = {
  '@context': 'https://schema.org',
  '@type': 'WebSite',
  name: 'DebateLab',
  url: BASE_URL,
  description: 'Practice debating with an AI opponent. Get instant feedback and improve your debate skills.',
  potentialAction: {
    '@type': 'SearchAction',
    target: {
      '@type': 'EntryPoint',
      urlTemplate: `${BASE_URL}/debate?topic={search_term_string}`,
    },
    'query-input': 'required name=search_term_string',
  },
}

export const softwareApplicationSchema = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'DebateLab',
  applicationCategory: 'EducationalApplication',
  operatingSystem: 'Web',
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'USD',
  },
  description: 'AI-powered debate practice platform with instant feedback and personalized coaching',
  url: BASE_URL,
}

// Breadcrumb structured data helper
export function breadcrumbSchema(items) {
  // items: [{ name: 'Home', url: '/' }, { name: 'Debate', url: '/debate' }]
  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: item.name,
      item: item.url.startsWith('http') ? item.url : `${BASE_URL}${item.url}`,
    })),
  }
}

// FAQ structured data
export const faqSchema = {
  '@context': 'https://schema.org',
  '@type': 'FAQPage',
  mainEntity: [
    {
      '@type': 'Question',
      name: 'What is DebateLab?',
      acceptedAnswer: {
        '@type': 'Answer',
        text: 'DebateLab is an AI-powered debate practice platform that allows you to practice debating with an intelligent AI opponent. Get instant feedback, improve your argumentation skills, and master debate techniques with personalized coaching.',
      },
    },
    {
      '@type': 'Question',
      name: 'How does DebateLab work?',
      acceptedAnswer: {
        '@type': 'Answer',
        text: 'You can start a debate by choosing a topic and your position (for or against). The AI opponent will take the opposite position and engage in a structured debate with you. After the debate, you receive detailed feedback and scores on your performance.',
      },
    },
    {
      '@type': 'Question',
      name: 'What debate topics can I practice?',
      acceptedAnswer: {
        '@type': 'Answer',
        text: 'DebateLab offers topics across four categories: Politics, Economics, Social issues, and Technology. You can choose from pre-generated topics or create your own custom debate topic.',
      },
    },
    {
      '@type': 'Question',
      name: 'Is DebateLab free?',
      acceptedAnswer: {
        '@type': 'Answer',
        text: 'Yes, DebateLab is completely free to use. Practice as many debates as you want and improve your skills at no cost.',
      },
    },
    {
      '@type': 'Question',
      name: 'What feedback do I get after a debate?',
      acceptedAnswer: {
        '@type': 'Answer',
        text: 'After completing a debate, you receive a comprehensive score (0-10) with detailed feedback on Content & Structure, Engagement & Clash, and Strategy & Execution. The AI also identifies your weaknesses and recommends targeted drills to improve.',
      },
    },
  ],
}

// HowTo structured data for debate practice
export const howToDebateSchema = {
  '@context': 'https://schema.org',
  '@type': 'HowTo',
  name: 'How to Practice Debating with DebateLab',
  description: 'Step-by-step guide to practicing and improving your debate skills using DebateLab',
  step: [
    {
      '@type': 'HowToStep',
      position: 1,
      name: 'Choose a Debate Topic',
      text: 'Select a topic from Politics, Economics, Social, or Technology categories, or create your own custom topic.',
    },
    {
      '@type': 'HowToStep',
      position: 2,
      name: 'Choose Your Position',
      text: 'Decide whether you want to argue for or against the topic. The AI will take the opposite position.',
    },
    {
      '@type': 'HowToStep',
      position: 3,
      name: 'Start the Debate',
      text: 'Begin the debate by making your opening argument. The AI will respond, and you can continue the debate for multiple rounds.',
    },
    {
      '@type': 'HowToStep',
      position: 4,
      name: 'Receive Feedback',
      text: 'After completing the debate, review your score and detailed feedback on your performance, including areas for improvement.',
    },
    {
      '@type': 'HowToStep',
      position: 5,
      name: 'Practice Targeted Drills',
      text: 'Use the recommended drills to focus on specific weaknesses like rebuttal, structure, weighing, evidence, or strategy.',
    },
  ],
}

