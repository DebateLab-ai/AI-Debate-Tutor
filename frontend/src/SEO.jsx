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

