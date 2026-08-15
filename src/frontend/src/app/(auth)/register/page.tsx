'use client'

import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import RegisterForm from '@/app/(auth)/register/RegisterForm'
import React from 'react'

const RegisterPage = () => {
  const { status } = useSession()
  const router = useRouter()

  useEffect(() => {
    if (status === 'authenticated') {
      router.replace('/')
    }
  }, [router, status])

  if (status === 'loading') {
    return null
  }
  return (
    <div>
      <RegisterForm />
    </div>
  )
}

export default RegisterPage
